import torch
import torch.nn.functional as F
import wandb
import os, time
from tqdm import tqdm
import torch.distributed as dist

# Helix2Text modules
from .contrastive import DNATextContrastiveLoss
from .evaluation import evaluate_intrinsic, evaluate_extrinsic


def train_model(config=None):
    """Main training function for DNA-Text fusion models."""

    # Load configuration
    model_engine = config.deepspeed_runtime['model_engine']
    device = config.deepspeed_runtime['device']
    local_rank = config.deepspeed_runtime.get('rank', -1)

    # Fetch the tokenizer from the config
    dna_tokenizer = config.deepspeed_runtime['dna_tokenizer']
    text_tokenizer = config.deepspeed_runtime['text_tokenizer']

    # Fetch optimizer
    optimizer = model_engine.optimizer

    # Fetch data from model_engine
    train_loader = config.deepspeed_runtime['train_loader']
    test_loader = config.deepspeed_runtime['test_loader']
    inference_loader = config.deepspeed_runtime['inference_loader']

    # Only initialize wandb on rank 0
    if local_rank <= 0:
        wandb.init(
            project="DNA-Text-Fusion",
            config={
                "batch_size": config.training.batch_size,
                "epochs": config.training.num_epochs,
                "learning_rate": config.training.learning_rate,
                "grad_accum_steps": config.training.grad_accum_steps,
                "train_data_fraction": config.training.train_data_fraction,
                "deepspeed": True,
                "local_rank": local_rank,
                "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            }
        )


    # Initialize loss
    contrastive_loss_fn = DNATextContrastiveLoss(
        initial_temp=0.5,
        min_temp=0.07,
        total_epochs=config.training.num_epochs,
    )
    
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    
    #=======================================================================================================
    #                                                                                          Training Loop
    #=======================================================================================================

    # Training tracking variables
    
    # Start training loop
    for epoch in range(config.training.num_epochs):
            
        # Set model to training mode
        model_engine.train()

        # Initialize variables
        max_step_times = 100
        epoch_loss = 0.0
        step_times = []

        #progress bar for training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}") if local_rank <= 0 else train_loader

        # Training steps
        for step, batch in enumerate(progress_bar):
            
            # log time
            step_start_time = time.time()
            steps_done = step + 1 + epoch * len(train_loader)
            total_training_steps = config.training.num_epochs * len(train_loader)

            # Move inputs to device - handle list structure for DNA inputs
            dna_inputs = {
                "dna_input_ids": batch["dna_input_ids"],  # List of lists of tensors
            }
            text_inputs = {
                "input_ids": batch["text_input_ids"].to(device),
            }
            
            outputs = model_engine(
                dna_inputs=dna_inputs, 
                text_inputs=text_inputs, 
                dna_mask_positions=batch.get("dna_mask_positions", None)
            )

            # Proper bidirectional loss computation with target sequences
            text_targets = batch.get("text_target_ids", batch["text_input_ids"]).to(device)
            
            # Cross-entropy loss for Text generation
            # Shift targets for autoregressive prediction
            text_logits_shifted = outputs["text_logits"][:, :-1, :].contiguous()
            text_targets_shifted = text_targets[:, 1:].contiguous()
            text_ce_loss = ce_loss_fn(
                text_logits_shifted.view(-1, text_logits_shifted.size(-1)),
                text_targets_shifted.view(-1)
            )

            # DNA generation loss - fully vectorized computation
            dna_ce_loss = torch.tensor(0.0, device=device)
            
            # Use DNA positions from model output (already calculated efficiently)
            if "dna_logits" in outputs and outputs["dna_logits"] is not None and outputs["dna_positions"] is not None:
                dna_logits = outputs["dna_logits"]  # [batch_size, text_seq_len, dna_vocab_size]
                dna_positions = outputs["dna_positions"]  # List of positions per batch
                
                # Collect all valid positions and targets for batch processing
                all_logits = []
                all_targets = []
                
                for batch_idx, positions in enumerate(dna_positions):
                    if len(positions) > 0 and batch_idx < len(batch["dna_input_ids"]):
                        sample_dna_sequences = batch["dna_input_ids"][batch_idx]
                        
                        for pos_idx, pos in enumerate(positions):
                            if (pos_idx < len(sample_dna_sequences) and 
                                pos < dna_logits.size(1) and 
                                len(sample_dna_sequences[pos_idx]) > 0):
                                
                                dna_sequence = sample_dna_sequences[pos_idx]
                                sequence_length = min(len(dna_sequence), dna_logits.size(1) - pos)
                                
                                # Vectorized sequence processing
                                if sequence_length > 0:
                                    # Extract logits for the entire sequence at once
                                    end_pos = min(pos + sequence_length, dna_logits.size(1))
                                    sequence_logits = dna_logits[batch_idx, pos:end_pos, :]  # [seq_len, vocab_size]
                                    sequence_targets = dna_sequence[:end_pos-pos].to(device)  # [seq_len]
                                    
                                    # Add to batch tensors
                                    all_logits.append(sequence_logits)
                                    all_targets.append(sequence_targets)
                
                # Compute all losses at once if we have any data
                if all_logits and all_targets:
                    # Concatenate all sequences
                    batch_logits = torch.cat(all_logits, dim=0)  # [total_seq_len, vocab_size]
                    batch_targets = torch.cat(all_targets, dim=0)  # [total_seq_len]
                    
                    # Single vectorized loss computation for all sequences
                    dna_ce_loss = ce_loss_fn(batch_logits, batch_targets)
            
            # Contrastive loss between embeddings
            contrastive_loss, _ = contrastive_loss_fn(
                outputs["joint_embedding"],
                outputs["fused_representation"],
                epoch=epoch
            )
            
            # Additional sequence-level losses using pooled outputs
            sequence_level_loss = torch.tensor(0.0, device=device)
            if "text_pooled" in outputs and "dna_pooled" in outputs:
                # Add L2 regularization on pooled representations to prevent collapse
                text_pooled_norm = torch.norm(outputs["text_pooled"], p=2, dim=-1).mean()
                dna_pooled_norm = torch.norm(outputs["dna_pooled"], p=2, dim=-1).mean()
                
                # Encourage diversity in pooled representations (prevent mode collapse)
                pooled_diversity_loss = -torch.log(text_pooled_norm + 1e-8) - torch.log(dna_pooled_norm + 1e-8)
                sequence_level_loss = 0.01 * pooled_diversity_loss  # Small weight
            
            # Vectorized retrieval consistency loss using individual DNA contexts
            retrieval_loss = torch.tensor(0.0, device=device)
            if outputs.get("individual_dna_contexts"):
                # Collect all context pairs for batch processing
                all_ctx1, all_ctx2 = [], []
                
                for contexts in outputs["individual_dna_contexts"]:
                    if len(contexts) > 1:  # Need at least 2 contexts for comparison
                        # Convert contexts to tensors and pool
                        pooled_contexts = []
                        for ctx in contexts:
                            if len(ctx) > 0:
                                pooled_ctx = ctx.mean(dim=0) if len(ctx.shape) > 1 else ctx
                                pooled_contexts.append(pooled_ctx)
                        
                        # Create all pairwise combinations
                        if len(pooled_contexts) > 1:
                            for i in range(len(pooled_contexts)):
                                for j in range(i+1, len(pooled_contexts)):
                                    all_ctx1.append(pooled_contexts[i])
                                    all_ctx2.append(pooled_contexts[j])
                
                # Vectorized similarity computation if we have pairs
                if all_ctx1 and all_ctx2:
                    ctx1_batch = torch.stack(all_ctx1)  # [num_pairs, hidden_dim]
                    ctx2_batch = torch.stack(all_ctx2)  # [num_pairs, hidden_dim]
                    
                    # Vectorized cosine similarity
                    similarities = F.cosine_similarity(ctx1_batch, ctx2_batch, dim=1)  # [num_pairs]
                    
                    # Target similarity and MSE loss computation
                    target_similarities = torch.full_like(similarities, 0.5)
                    retrieval_consistency = F.mse_loss(similarities, target_similarities)
                    retrieval_loss = 0.005 * retrieval_consistency  # Very small weight

            # Combine losses with weights
            # Scale DNA loss based on whether it was computed
            dna_weight = 1.0 if dna_ce_loss.item() > 0.0 else 0.0
            total_loss = (
                text_ce_loss +
                dna_weight * dna_ce_loss +
                #contrastive_loss +
                sequence_level_loss +
                retrieval_loss
            )
            
            # Scale loss for gradient accumulation
            scaled_loss = total_loss / config.training.grad_accum_steps
            
            # DeepSpeed backward pass
            model_engine.backward(scaled_loss)
            model_engine.step()

            # Track loss (use unscaled loss for logging)
            epoch_loss += total_loss.item()
            
            # Calculate ETA
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time

            # Update step times list
            step_times.append(step_duration)
            if len(step_times) > max_step_times:
                step_times.pop(0)  # Remove oldest timing

            # Calculate ETA based on recent average step time
            avg_step_time = sum(step_times) / len(step_times)
            total_training_steps = config.training.num_epochs * len(train_loader)
            steps_done = (epoch * len(train_loader)) + step + 1
            remaining_steps = total_training_steps - steps_done
            eta_seconds = remaining_steps * avg_step_time
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            
            # Format ETA for display
            if eta_hours > 1:
                eta_display = f"{eta_hours:.1f}h"
            else:
                eta_display = f"{eta_minutes:.1f}m"

            # Log step metrics
            if local_rank <= 0 and wandb.run:
                wandb.log({
                "loss/total_loss": total_loss.item(),
                "loss/DNA": dna_ce_loss.item(),
                "loss/Text": text_ce_loss.item(),
                "loss/Contrastive": contrastive_loss.item(),
                "loss/Sequence_Level": sequence_level_loss.item(),
                "loss/Retrieval": retrieval_loss.item(),
                "training/learning_rate": optimizer.param_groups[0]['lr'],
                "training/step": steps_done,
                "training/eta_hours": eta_hours,
                })  
            
            
            # Add indicator for gradient accumulation progress
            accum_step = (step % config.training.grad_accum_steps) + 1
            accum_percent = accum_step / config.training.grad_accum_steps * 100
            
            # Update progress bar
            if local_rank <= 0:
                accum_step = (step % config.training.grad_accum_steps) + 1
                progress_bar.set_postfix({
                    "Loss": f"{total_loss.item():.4f}",
                    "DNA": f"{dna_ce_loss.item():.4f}",
                    "Text": f"{text_ce_loss.item():.4f}",
                    "Cont": f"{contrastive_loss.item():.4f}",
                    "Seq": f"{sequence_level_loss.item():.4f}",
                    "Ret": f"{retrieval_loss.item():.4f}",
                    "Accum": f"{accum_step}/{config.training.grad_accum_steps} ({accum_percent:.0f}%)",
                    "ETA": eta_display,
                    "Progress": f"{steps_done}/{total_training_steps} ({(steps_done/total_training_steps)*100:.1f}%)"
                })
                

        # Set epoch for distributed sampler to ensure proper shuffling
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Comprehensive evaluation with intrinsic and extrinsic metrics
        if local_rank <= 0:
            eval_model = model_engine.module
            
            # Get DNA lookup from config if available
            dna_lookup = getattr(config, 'dna_lookup', None)
            
            # Intrinsic evaluation (fast, per epoch)
            intrinsic_metrics = evaluate_intrinsic(
                model=eval_model,
                test_loader=test_loader,
                text_tokenizer=text_tokenizer,
                dna_tokenizer=dna_tokenizer,
                device=device,
                dna_lookup=dna_lookup
            )
            
            # Log epoch metrics
            avg_loss = epoch_loss / len(train_loader)
            epoch_metrics = {
                "epoch/epoch": epoch + 1,
                "epoch/avg_loss": avg_loss,
            }
            
            # Add intrinsic metrics with prefix
            for key, value in intrinsic_metrics.items():
                epoch_metrics[f"intrinsic/{key}"] = value
            
            
            wandb.log(epoch_metrics)
            
            # Print summary
            print(f"\n📊 Epoch {epoch+1} Evaluation:")
            print("Intrinsic metrics:")
            if "bleu" in intrinsic_metrics:
                print(f"  Text BLEU: {intrinsic_metrics.get('bleu', 0):.4f}")
            if "text_loss" in intrinsic_metrics:
                print(f"  Text Loss: {intrinsic_metrics.get('text_loss', 0):.4f}")
            if "dna_per_base_accuracy" in intrinsic_metrics:
                print(f"  DNA Accuracy: {intrinsic_metrics.get('dna_per_base_accuracy', 0):.4f}")

        else:
            intrinsic_metrics = {}

        
        # Save model checkpoint every 10 epochs
        '''
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config.training.num_epochs:
            print(f"💾 Saving checkpoint at epoch {epoch + 1}")
            if hasattr(model_engine, 'save_checkpoint'):
                if dist.is_initialized():
                    dist.barrier()

                # Ensure the checkpoint directory exists
                os.makedirs("./checkpoints", exist_ok=True)
                model_engine.save_checkpoint("./checkpoints")
        '''

    
    eval_model = model_engine.module

    
    if local_rank <= 0:
        # Final comprehensive evaluation
        print("🔬 Running final comprehensive evaluation...")
        
        # Get DNA lookup from config if available
        dna_lookup = getattr(config, 'dna_lookup', None)
        
        # Final intrinsic evaluation
        final_intrinsic = evaluate_intrinsic(
            model=model_engine.module,
            test_loader=test_loader,
            text_tokenizer=text_tokenizer,
            dna_tokenizer=dna_tokenizer,
            device=device,
            dna_lookup=dna_lookup
        )

        # Final extrinsic evaluation 
        final_extrinsic = evaluate_extrinsic(
            model=model_engine.module,
            inference_loader=inference_loader,
            text_tokenizer=text_tokenizer,
            dna_tokenizer=dna_tokenizer,
            device=device,
            num_samples=5 # More samples for final evaluation
        )
        
        # Combine final metrics
        final_metrics = {**final_intrinsic, **final_extrinsic}

        # Log final metrics to wandb
        final_wandb_metrics = {
            "final/intrinsic_bleu": final_intrinsic.get("bleu", 0.0),
            "final/intrinsic_rouge1": final_intrinsic.get("rouge1", 0.0),
            "final/intrinsic_rougeL": final_intrinsic.get("rougeL", 0.0),
            "final/intrinsic_text_loss": final_intrinsic.get("text_loss", 0.0),
            "final/extrinsic_bleu": final_extrinsic.get("full_output_bleu", 0.0),
            "final/joint_coherence": final_extrinsic.get("joint_coherence", 0.0),
            "final/avg_dna_insertions": final_extrinsic.get("avg_dna_insertions_per_sample", 0.0),
            "training_completed": True
        }
        
        # Add DNA metrics if available
        if "dna_per_base_accuracy" in final_intrinsic:
            final_wandb_metrics["final/dna_per_base_accuracy"] = final_intrinsic["dna_per_base_accuracy"]
        if "dna_edit_distance" in final_intrinsic:
            final_wandb_metrics["final/dna_edit_distance"] = final_intrinsic["dna_edit_distance"]
        
        # Log final output examples as a wandb table if available
        if "output_examples" in final_extrinsic and final_extrinsic["output_examples"]:
            final_examples_table = wandb.Table(
                columns=["Sample ID", "Reference Text", "Generated Text", "DNA Insertions", "DNA Sequences", "Coherence Score"],
                data=[[ex['sample_id'], ex['reference_text'], ex['generated_text'], 
                      ex['num_dna_insertions'], ex['dna_sequences'], ex['coherence_score']] 
                     for ex in final_extrinsic["output_examples"]]
            )
            final_wandb_metrics["final/generation_examples"] = final_examples_table
        
        wandb.log(final_wandb_metrics)
        if wandb.run:
            wandb.finish()
    
        print(f"🎯 Final Results:")
        print(f"📝 Intrinsic Evaluation (Ground Truth Comparison):")
        print(f"  Text BLEU: {final_intrinsic.get('bleu', 0.0):.4f}")
        print(f"  Text ROUGE-L: {final_intrinsic.get('rougeL', 0.0):.4f}")
        print(f"  Text Loss: {final_intrinsic.get('text_loss', 0.0):.4f}")
        if "dna_per_base_accuracy" in final_intrinsic:
            print(f"  DNA Per-base Accuracy: {final_intrinsic.get('dna_per_base_accuracy', 0.0):.4f}")
        if "dna_edit_distance" in final_intrinsic:
            print(f"  DNA Edit Distance: {final_intrinsic.get('dna_edit_distance', 0.0):.4f}")
        
        print(f"🧬 Extrinsic Evaluation (Generation Quality):")
        print(f"  Full Output BLEU: {final_extrinsic.get('full_output_bleu', 0.0):.4f}")
        print(f"  Joint Coherence: {final_extrinsic.get('joint_coherence', 0.0):.4f}")
        print(f"  Avg DNA Insertions/Sample: {final_extrinsic.get('avg_dna_insertions_per_sample', 0.0):.2f}")
        
        return eval_model, final_metrics
    
    return model_engine.module, {}

