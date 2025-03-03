import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os
import numpy as np
import json
import argparse
import time
import logging
from typing import Annotated, Dict, List, Optional, cast
from tqdm import tqdm
import math
import concurrent.futures
import multiprocessing  # Add this import

from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.retrievers.registry_utils import load_vision_retriever_from_registry

def recall_at_n(filtered_images, pos_images, N):
    """
    计算 Recall@N
    :param filtered_images: 模型预测的图片列表
    :param pos_images: ground-truth 图片列表
    :param N: 考虑的前 N 个预测
    :return: recall@N 的值
    """
    # 截取前 N 个预测图片
    top_n_predictions = filtered_images[:N]
    
    # 计算匹配的数量
    matches = sum(1 for img in top_n_predictions if img in pos_images)
    
    # 计算 recall@N
    recall = matches / len(pos_images) if len(pos_images) > 0 else 0
    return recall

# Ensure that the class is defined at the top-level scope, not inside a function
class CoPaliSolver:
    def __init__(self, vision_retriever, image_root='data/Test'):
        self.image_root = image_root
        self.vision_retriever = vision_retriever
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 8

    def construct_vision_embeddings(self, image_dir, doc_id, batch_size=8):
        """
        Constructs the vision embeddings for the images in the given directory.
        """
        start_time = time.time()
        image_list = os.listdir(image_dir)
        image_list = [os.path.join(image_dir, x) for x in image_list if doc_id in x]
        image_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        
        images = [Image.open(x) for x in image_list]
        
        # Prebatching images to optimize memory usage
        emb_images = []
        dataloader_prebatch_size = 10 * batch_size

        for passage_batch in tqdm(
            batched(images, n=dataloader_prebatch_size),
            desc="Dataloader pre-batching",
            total=math.ceil(len(images) / (dataloader_prebatch_size)),
        ):
            batch_emb_passages = self.vision_retriever.forward_passages(passage_batch, batch_size=batch_size)
            if isinstance(batch_emb_passages, torch.Tensor):
                batch_emb_passages = list(torch.unbind(batch_emb_passages))
                emb_images.extend(batch_emb_passages)
            else:
                emb_images.extend(batch_emb_passages)

        elapsed_time = time.time() - start_time
        return emb_images, image_list, elapsed_time

    def get_combined_top_k_images(self, needle_word, emb_images, image_files, k=10):
        """
        Retrieves the top-k images based on the query.
        """
        start_time = time.time()
        emb_queries = self.vision_retriever.forward_queries([needle_word], batch_size=1)
        scores = self.vision_retriever.get_scores(emb_queries, emb_images, batch_size=1)[0]
        score_dict = {file: score for file, score in zip(image_files, scores)}

        top_k_images = sorted(score_dict.keys(), key=lambda x: score_dict[x], reverse=True)[:k]

        elapsed_time = time.time() - start_time
        return top_k_images, elapsed_time

    def process_entry(self, entry, output_dir, top_k_filter=30, use_filter=False):
        """
        Processes a single entry and returns the results for logging.
        """
        question = entry["question"]
        id = entry["question_id"]
        image_dir = os.path.join(self.image_root, entry["doc_no"][:4])  # 4 digits

        emb_images, image_files, construct_time = self.construct_vision_embeddings(image_dir, entry["doc_no"])
        top_k_images, combined_time = self.get_combined_top_k_images(question, emb_images, image_files, k=top_k_filter)

        if use_filter:
            filtered_images, llava_filtering_time = self.filter_with_llava(top_k_images, question)
        else:
            filtered_images = top_k_images
            llava_filtering_time = 0

        filtered_images = [img.split("/")[-1] for img in filtered_images]
        pos_image_ids = entry["evidence_pages"]
        pos_images = [image_files[x-1].split("/")[-1] for x in pos_image_ids]

        # Recall
        recall_dict = {}
        N = [5, 10, 20, 30]
        for i, n in enumerate(N):
            recall = recall_at_n(filtered_images, pos_images, n)
            recall_dict[n] = recall

        file_name = id + ".json"
        output_file = os.path.join(output_dir, file_name)

        pos_images = [pos_image.split(".")[0] for pos_image in pos_images]
        with open(output_file, "w") as f_out:
            json.dump({
                "question": question,
                "top_10_images": filtered_images[:10],
                "recall_at_n": recall_dict,
                "real_positive_image": pos_images
            }, f_out, indent=4)

        return combined_time + construct_time, llava_filtering_time, recall_dict

    def process_dataset(self, dataset_file, output_dir, top_k_filter=30, use_filter=False):
        """
        Processes the entire dataset in parallel.
        """
        with open(dataset_file, 'r') as file:
            data = [json.loads(line) for line in file]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        total_entries = len(data)
        total_recall_at_n = [0, 0, 0, 0]
        N = [5, 10, 20, 30]
        total_combined_time = 0
        total_llava_filtering_time = 0

        # Using multiprocessing for parallel evaluation
        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
            futures = []
            for entry in data:
                futures.append(executor.submit(self.process_entry, entry, output_dir, top_k_filter, use_filter))

            for future in tqdm(concurrent.futures.as_completed(futures), desc="Processing entries", total=total_entries):
                combined_time, llava_filtering_time, recall_dict = future.result()
                total_combined_time += combined_time
                total_llava_filtering_time += llava_filtering_time

                for i, n in enumerate(N):
                    total_recall_at_n[i] += recall_dict[n]

        total_recall_at_n = [x / total_entries for x in total_recall_at_n]
        avg_combined_time = (total_combined_time) / total_entries
        avg_llava_filtering_time = total_llava_filtering_time / total_entries

        log_file = os.path.join(output_dir, 'recall.log')

        with open(log_file, 'w') as log:
            log.write(f"Total Entries: {total_entries}\n")
            for i, n in enumerate(N):
                log.write(f"Top-{n} Recall: {total_recall_at_n[i]:.2%}\n")
            log.write(f"Average Combined Encoder Inference Time: {avg_combined_time:.4f} seconds\n")
            log.write(f"Average LLaVA Filtering Time: {avg_llava_filtering_time:.4f} seconds\n")

        print(f"Total Entries: {total_entries}")
        for i, n in enumerate(N):
            print(f"Top-{n} Recall: {total_recall_at_n[i]:.2%}")
        print(f"Average Combined Encoder Inference Time: {avg_combined_time:.4f} seconds")
        print(f"Average LLaVA Filtering Time: {avg_llava_filtering_time:.4f} seconds")


# Entry point for the script
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Run CoPali Solver with adjustable dataset and image directory.")
    parser.add_argument('--dataset_file', type=str, default="data/LongDocURL_public.jsonl", help="Path to the dataset JSONL file")
    parser.add_argument('--image_root', type=str, default="data/pdf_pngs", help="Root directory for images")
    parser.add_argument('--top_k_filter', type=int, default=30, help="Top K filter for image retrieval") # used in v-rag
    parser.add_argument('--use_filter', action='store_true', help="Use LLaVA to filter irrelevant images")
    parser.add_argument('--output_dir', type=str, default="retrieval_results/copali_results", help="Output directory for results")
    parser.add_argument('--model_class', type=str, default="colqwen2", help="Model class for the retriever")
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='Path to the pretrained model')
    args = parser.parse_args()

    retriever = load_vision_retriever_from_registry(
        args.model_class,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
    )

    solver = CoPaliSolver(vision_retriever=retriever, image_root=args.image_root)
    solver.process_dataset(args.dataset_file, args.output_dir, top_k_filter=args.top_k_filter, use_filter=args.use_filter)
