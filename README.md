# Text-to-CAD

This repository contains all work done towards a physics-aware text to CAD generation model. Each folder has the following contents:
*  **CADFusion-Benchmark**: Implemented Microsoft's LoRA finetuned CADFusion model for a baseline of existing text to CAD models. The aim was to test a finetuned model that generated CAD sequence tokens directly as a baseline. The results demonstrated poor performance due to lacking a feedback loop and direct CAD sequence generation rather than scripting (e.g. CADQuery). Because of this, we tested larger base models and their capabilites with generating CADQuery when assisted by a simple feedback loop.
  
*  **Gemma3-12B Benchmark Pipeline**: Created a generation pipeline for a Gemma3-12B, a larger parameter open source model, aiming to compare larger base models with smaller finetuned models. This pipeline used a simple feedback loop that integrated 5 retries with previous CADQuery compilation errors and STL conversion errors fed back to the LLM. There was much more success when compared to CADFusion, so the decision was made to choose a larger base model to integrate physics into for the best performance.

*  **Gemma3-12B PiSSA Finetune**: Outlines the process and setup used to attempt finetuning Gemma3-12B using PiSSA, a modern, lightweight finetuning method capable of accurately finetuning large models quickly.  The goal was to test a modern fine tuning method and its ability to improve a model in CADQuery generation. Ultimately, the finetune failed due to C++ backend errors in Llama Factory. Now, I am transitioning to testing DoRA, another highly accurate and efficient way to fine tune a large parameter LLM.

Currently exploring finetuning methods to attempt injecting physics awareness into a larger parameter base model.
