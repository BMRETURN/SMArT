

## **📌 SMArT**

Code release for **SMArT: A Self-driven Multi-view Architecture Leveraging Large Language Models for Time Series Forecastin**

## **🔍 Abstract**

Time series forecasting (TSF) is critical in numerous real-world applications, yet its sequential scalar presentation limits semantic richness and the capture of complex temporal patterns. Recent advances leveraging patch-wise modeling and pre-trained large language models (LLMs) have achieved notable progress. However, existing methods largely focus on raw sequential patterns while overlooking intra-patch semantics, which limits the richness of time-series representations, and they fail to effectively exploit complementary information from multiple views. To tackle these challenges, we propose Self-driven Multi-view Architecture for TSF, named **SMArT.** SMArT enriches intra-patch semantics through self-supervised multi-view fusion. It jointly captures fine-grained temporal 1D dependencies via point-wise self-attention and global temporal 2D structures via Gramian Angular Fields, all without requiring external supervision. To further bridge the gap between time series data and LLMs, SMArT pioneers a dual-prompt strategy, combining static, dataset-level guidance with dynamic, input-specific prompts derived from frequency domain decomposition, significantly enhancing LLMs' adaptability and generalization. Extensive experiments across diverse TSF tasks validate SMArT’s robustness, achieving state-of-the-art performance in long-term forecasting and excelling in few-shot and zero-shot scenarios. By integrating self-driven multi-view learning with LLMs’ reasoning power, SMArT establishes an effective framework for time series forecasting.

## ** 🧬Key Features**

- **Self-driven multi-view learning** — automatically generate multiple complementary views from time series.  
- **LLM alignment** — adapt learned representations into LLM embedding space for richer semantics.  
- **Modular & extensible** — easy to plug in new datasets, views, or forecasting models.  
- **End-to-end training + evaluation** pipeline.

## **⚙️ Start**

1. pip install -r requirements.txt
2. Obtain the well pre-processed datasets from [Time-Series-Library](https://github.com/thuml/Time-Series-Library).  Then place the downloaded data in the folder: `./dataset`.
3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`.
