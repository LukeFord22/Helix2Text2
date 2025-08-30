"""Refactored evaluation metrics for DNA-text fusion models with intrinsic and extrinsic evaluation."""

import torch
import re
import numpy as np
from collections import Counter
import editdistance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score


def evaluate_intrinsic(model, test_loader, text_tokenizer, dna_tokenizer, device, dna_lookup=None):
    """
    Intrinsic evaluation: Compare model predictions vs ground truth without autoregressive generation.
    
    Args:
        model: The DNATextFusion model
        test_loader: DataLoader with test data (ground truth)
        text_tokenizer: Text tokenizer
        dna_tokenizer: DNA tokenizer  
        device: Device for computation
        dna_lookup: DNA part lookup dictionary
        
    Returns:
        Dictionary of intrinsic metrics
    """
    model.eval()
    
    # Text metrics
    text_predictions = []
    text_references = []
    text_losses = []
    
    # DNA metrics
    dna_predictions = []
    dna_references = []
    dna_losses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move batch to device
            text_inputs = {
                "input_ids": batch["text_input_ids"].to(device),
            }
            
            # Handle DNA inputs - get from lookup if available
            dna_inputs = None
            if dna_lookup and "dna_part_ids" in batch:
                dna_input_ids_batch = []
                
                for sample_parts in batch["dna_part_ids"]:
                    sample_dna_ids = []
                    
                    for part_id in sample_parts:
                        if part_id in dna_lookup:
                            dna_data = dna_lookup[part_id]
                            sample_dna_ids.append(dna_data['input_ids'])
                    
                    dna_input_ids_batch.append(sample_dna_ids)
                
                if dna_input_ids_batch and dna_input_ids_batch[0]:
                    dna_inputs = {
                        "dna_input_ids": dna_input_ids_batch,
                    }
            
            if dna_inputs is None:
                # Create dummy DNA inputs if none available
                batch_size = text_inputs["input_ids"].size(0)
                dna_inputs = {
                    "dna_input_ids": [[torch.zeros(10, dtype=torch.long, device=device)] for _ in range(batch_size)],
                }
            
            # Forward pass - flash attention handles masking natively
            outputs = model(dna_inputs=dna_inputs, text_inputs=text_inputs)
            
            # Text evaluation
            text_logits = outputs["text_logits"]  # [batch, seq_len, vocab_size]
            text_targets = text_inputs["input_ids"]  # Ground truth
            
            # Calculate text loss (shifted for autoregressive prediction)
            text_logits_shifted = text_logits[:, :-1, :].contiguous()
            text_targets_shifted = text_targets[:, 1:].contiguous()
            text_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            
            text_loss = text_loss_fn(
                text_logits_shifted.view(-1, text_logits_shifted.size(-1)),
                text_targets_shifted.view(-1)
            )
            
            # Calculate text loss - flash attention handles masking natively
            # Use non-padding tokens (assuming padding token is 0)
            valid_mask = (text_targets_shifted != 0).float()
            text_loss = text_loss * valid_mask.view(-1)
            valid_tokens = valid_mask.sum().item()
            
            if valid_tokens > 0:
                text_losses.append(text_loss.sum().item() / valid_tokens)
            
            # Get text predictions and references for BLEU/ROUGE/METEOR
            text_pred_ids = torch.argmax(text_logits_shifted, dim=-1)
            
            for i in range(text_pred_ids.size(0)):
                # Get non-padded sequences based on non-zero tokens
                valid_tokens_mask = (text_targets_shifted[i] != 0)
                if valid_tokens_mask.sum().item() > 0:
                    pred_sequence = text_pred_ids[i][valid_tokens_mask]
                    ref_sequence = text_targets_shifted[i][valid_tokens_mask]
                    
                    pred_text = text_tokenizer.decode(pred_sequence, skip_special_tokens=True)
                    ref_text = text_tokenizer.decode(ref_sequence, skip_special_tokens=True)
                    
                    text_predictions.append(pred_text)
                    text_references.append(ref_text)
            
            # DNA evaluation (if DNA data available)
            if "dna_logits" in outputs and outputs["dna_logits"] is not None:
                dna_logits = outputs["dna_logits"]  # [batch, text_seq_len, dna_vocab_size]
                
                # Find DNA mask positions for evaluation
                DNA_MASK_TOKEN = 200010  # <|reserved_200010|>
                dna_mask_positions = (text_targets == DNA_MASK_TOKEN).nonzero(as_tuple=False)
                
                if len(dna_mask_positions) > 0:
                    for batch_idx, pos_idx in dna_mask_positions:
                        batch_idx, pos_idx = batch_idx.item(), pos_idx.item()
                        
                        # Get corresponding DNA reference from dna_part_ids
                        if (batch_idx < len(batch["dna_part_ids"]) and 
                            len(batch["dna_part_ids"][batch_idx]) > 0):
                            
                            part_id = batch["dna_part_ids"][batch_idx][0]  # Take first part
                            if part_id in dna_lookup:
                                dna_ref = dna_lookup[part_id]['input_ids']
                                
                                # Get DNA prediction at this position
                                if pos_idx < dna_logits.size(1):
                                    dna_pred_logits = dna_logits[batch_idx, pos_idx, :]
                                    dna_pred_token = torch.argmax(dna_pred_logits).item()
                                    
                                    # Calculate DNA loss
                                    dna_target_token = dna_ref[0].item() if len(dna_ref) > 0 else 0
                                    dna_loss = torch.nn.functional.cross_entropy(
                                        dna_pred_logits.unsqueeze(0),
                                        torch.tensor([dna_target_token], device=device)
                                    )
                                    dna_losses.append(dna_loss.item())
                                    
                                    # Store for sequence-level metrics
                                    dna_pred_seq = dna_tokenizer.decode([dna_pred_token], skip_special_tokens=True)
                                    dna_ref_seq = dna_tokenizer.decode(dna_ref[:10], skip_special_tokens=True)  # First 10 tokens
                                    
                                    dna_predictions.append(dna_pred_seq)
                                    dna_references.append(dna_ref_seq)
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    # Calculate intrinsic metrics
    metrics = {}
    
    # Text metrics
    if text_predictions and text_references:
        metrics.update(_calculate_text_metrics(text_predictions, text_references))
    
    if text_losses:
        metrics["text_loss"] = np.mean(text_losses)
        metrics["text_perplexity"] = np.exp(metrics["text_loss"])
    
    # DNA metrics  
    if dna_predictions and dna_references:
        metrics.update(_calculate_dna_metrics(dna_predictions, dna_references))
    
    if dna_losses:
        metrics["dna_loss"] = np.mean(dna_losses)
        metrics["dna_perplexity"] = np.exp(metrics["dna_loss"])
    
    return metrics


def evaluate_extrinsic(model, inference_loader, text_tokenizer, dna_tokenizer, device, max_length=512, dna_max_length=128, num_samples=None):
    """
    Extrinsic evaluation: Full autoregressive generation with DNA routing and coherence checking.
    
    Args:
        model: The DNATextFusion model
        inference_loader: DataLoader with inference data
        text_tokenizer: Text tokenizer
        dna_tokenizer: DNA tokenizer
        device: Device for computation
        max_length: Maximum generation length
        dna_max_length: Maximum DNA sequence length
        num_samples: Number of samples to evaluate (None = all)
        
    Returns:
        Dictionary of extrinsic metrics
    """
    model.eval()
    
    generated_outputs = []
    reference_outputs = []
    dna_routing_stats = []
    coherence_scores = []
    
    # For wandb logging
    output_examples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(inference_loader):
            if num_samples and batch_idx >= num_samples:
                break
                
            # Get input context (partial text to continue from)
            text_inputs = {
                "input_ids": batch["text_input_ids"].to(device),
            }
            
            # Get DNA inputs for this batch (if available)
            batch_dna_inputs = None
            batch_dna_mask_positions = None
            
            if "dna_input_ids" in batch:
                batch_dna_inputs = {"dna_input_ids": batch["dna_input_ids"]}
                batch_dna_mask_positions = batch.get("dna_mask_positions", None)
            
            # Generate with unified routing
            generation_result = model.generate_unified(
                text_inputs=text_inputs,
                text_tokenizer=text_tokenizer,
                dna_tokenizer=dna_tokenizer,
                max_length=max_length,
                dna_max_length=dna_max_length,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                dna_inputs=batch_dna_inputs,
                dna_mask_positions=batch_dna_mask_positions
            )
            
            generated_ids = generation_result["generated_ids"]
            dna_events = generation_result["dna_routing_events"]
            
            # Process each sample in batch
            for i in range(generated_ids.size(0)):
                # Decode generated text
                generated_text = text_tokenizer.decode(generated_ids[i], skip_special_tokens=False)
                
                # Get reference text
                reference_text = text_tokenizer.decode(text_inputs["input_ids"][i], skip_special_tokens=False)
                
                generated_outputs.append(generated_text)
                reference_outputs.append(reference_text)
                
                # Check coherence (DNA start/end tags)
                coherence_score = _check_dna_coherence(generated_text)
                coherence_scores.append(coherence_score)
                
                # Track DNA routing statistics
                sample_dna_events = [event for event in dna_events if event['batch_idx'] == i]
                dna_sequences = [dna_tokenizer.decode(event['dna_sequence'][0], skip_special_tokens=False) 
                                for event in sample_dna_events]
                dna_routing_stats.append({
                    'num_dna_insertions': len(sample_dna_events),
                    'dna_sequences': dna_sequences
                })
                
                # Store example for wandb logging
                output_examples.append({
                    'sample_id': f'batch_{batch_idx}_sample_{i}',
                    'reference_text': reference_text,
                    'generated_text': generated_text,
                    'num_dna_insertions': len(sample_dna_events),
                    'dna_sequences': ', '.join(dna_sequences) if dna_sequences else 'None',
                    'coherence_score': coherence_score
                })
            
            # Memory cleanup
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
    
    # Calculate extrinsic metrics
    metrics = {}
    
    # Full output similarity metrics
    if generated_outputs and reference_outputs:
        metrics.update(_calculate_text_metrics(generated_outputs, reference_outputs, prefix="full_output_"))
    
    # DNA routing coherence
    if coherence_scores:
        metrics["joint_coherence"] = np.mean(coherence_scores)
    
    # DNA routing statistics  
    if dna_routing_stats:
        total_insertions = sum(stats['num_dna_insertions'] for stats in dna_routing_stats)
        avg_insertions = total_insertions / len(dna_routing_stats)
        metrics["avg_dna_insertions_per_sample"] = avg_insertions
        
        # DNA correctness (simplified - could be enhanced with reference comparison)
        all_dna_sequences = []
        for stats in dna_routing_stats:
            all_dna_sequences.extend(stats['dna_sequences'])
        
        if all_dna_sequences:
            avg_dna_length = np.mean([len(seq) for seq in all_dna_sequences])
            metrics["avg_generated_dna_length"] = avg_dna_length
    
    # Add output examples for wandb logging
    metrics["output_examples"] = output_examples
    
    return metrics


def _calculate_text_metrics(predictions, references, prefix=""):
    """Calculate text similarity metrics (BLEU, ROUGE, METEOR)."""
    metrics = {}
    
    # BLEU score
    try:
        smoothie = SmoothingFunction().method4
        bleu_scores = []
        for ref, pred in zip(references, predictions):
            try:
                score = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
                bleu_scores.append(score)
            except:
                pass
        
        if bleu_scores:
            metrics[f"{prefix}bleu"] = np.mean(bleu_scores)
    except Exception as e:
        print(f"BLEU calculation error: {e}")
    
    # ROUGE scores
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, pred in zip(references, predictions):
            try:
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            except:
                pass
        
        if rouge1_scores:
            metrics[f"{prefix}rouge1"] = np.mean(rouge1_scores)
            metrics[f"{prefix}rouge2"] = np.mean(rouge2_scores) 
            metrics[f"{prefix}rougeL"] = np.mean(rougeL_scores)
    except Exception as e:
        print(f"ROUGE calculation error: {e}")
    
    # METEOR score
    try:
        meteor_scores = []
        for ref, pred in zip(references, predictions):
            try:
                score = meteor_score([ref.split()], pred.split())
                meteor_scores.append(score)
            except:
                pass
        
        if meteor_scores:
            metrics[f"{prefix}meteor"] = np.mean(meteor_scores)
    except Exception as e:
        print(f"METEOR calculation error: {e}")
    
    return metrics


def _calculate_dna_metrics(predictions, references):
    """Calculate DNA-specific metrics (per-base accuracy, edit distance, k-mer similarity)."""
    metrics = {}
    
    # Per-base accuracy
    per_base_accuracies = []
    edit_distances = []
    kmer_similarities = []
    
    for pred, ref in zip(predictions, references):
        # Clean sequences (keep only ATGC)
        pred_clean = re.sub(r'[^ATGC]', '', pred.upper())
        ref_clean = re.sub(r'[^ATGC]', '', ref.upper())
        
        if len(pred_clean) > 0 and len(ref_clean) > 0:
            # Per-base accuracy
            min_len = min(len(pred_clean), len(ref_clean))
            if min_len > 0:
                matches = sum(p == r for p, r in zip(pred_clean[:min_len], ref_clean[:min_len]))
                accuracy = matches / min_len
                per_base_accuracies.append(accuracy)
            
            # Edit distance (normalized by max length)
            edit_dist = editdistance.eval(pred_clean, ref_clean)
            max_len = max(len(pred_clean), len(ref_clean))
            normalized_edit_dist = edit_dist / max_len if max_len > 0 else 1.0
            edit_distances.append(normalized_edit_dist)
            
            # K-mer similarity (3-mers)
            kmer_sim = _calculate_kmer_similarity(pred_clean, ref_clean, k=3)
            kmer_similarities.append(kmer_sim)
    
    if per_base_accuracies:
        metrics["dna_per_base_accuracy"] = np.mean(per_base_accuracies)
    
    if edit_distances:
        metrics["dna_edit_distance"] = np.mean(edit_distances)
    
    if kmer_similarities:
        metrics["dna_kmer_similarity"] = np.mean(kmer_similarities)
    
    return metrics


def _calculate_kmer_similarity(seq1, seq2, k=3):
    """Calculate k-mer similarity between two DNA sequences."""
    if len(seq1) < k or len(seq2) < k:
        return 0.0
    
    # Generate k-mers
    kmers1 = [seq1[i:i+k] for i in range(len(seq1) - k + 1)]
    kmers2 = [seq2[i:i+k] for i in range(len(seq2) - k + 1)]
    
    # Count k-mers
    counter1 = Counter(kmers1)
    counter2 = Counter(kmers2)
    
    # Calculate Jaccard similarity
    intersection = sum((counter1 & counter2).values())
    union = sum((counter1 | counter2).values())
    
    return intersection / union if union > 0 else 0.0


def _check_dna_coherence(text):
    """Check if DNA start/end tags are properly paired."""
    # Count DNA start and end tokens
    start_pattern = r'<\|reserved_200009\|>'  # <|reserved_200009|>
    end_pattern = r'<\|reserved_200011\|>'    # <|reserved_200011|>
    
    start_count = len(re.findall(start_pattern, text))
    end_count = len(re.findall(end_pattern, text))
    
    # Perfect coherence means equal start and end tags
    if start_count == 0 and end_count == 0:
        return 1.0  # No DNA tags, perfectly coherent
    
    if start_count == end_count:
        return 1.0  # Balanced tags
    
    # Partial coherence based on how close the counts are
    total_tags = start_count + end_count
    if total_tags == 0:
        return 1.0
    
    coherence = 1.0 - abs(start_count - end_count) / total_tags
    return max(0.0, coherence)

