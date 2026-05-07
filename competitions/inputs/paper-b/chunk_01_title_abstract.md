# Code Generation Competition: 16 Proprietary vs. Open-Source LLMs & Iterative Learning Based on FDA Adverse Event Reporting System

**Author:** Kevin Kawchak  
Chief Executive Officer  
ChemicalQDevice  
San Diego, CA  
December 22, 2025  
kevink@chemicalqdevice.com

---

## Abstract

Few effective goal-oriented iterative LLM code benchmarking studies exist. Successive high dimensional and complex problem improvements are desired versus conventional code assessments. Inspired by a recent CodeClash study, this tournament focuses primarily on the goal of generating functions to obtain a perfect competition task score based on three recent FDA FAERS files. Here, Opus 4.5 Extended was primarily utilized to build a novel Python evaluation engine measuring LLM code pair correctness, methodology, code quality, and algorithm effectiveness against a fixed reference standard and head-to-head. The notebook then automated Code A and Code B grading, and outputted their answers and reference standard of drug-reaction signals in csv files. The bracket was organized at scale: 16 LLMs - 8 proprietary LLMs on the left and 8 open-source LLMs on the right. The 8 Round 1 winners and corresponding notebooks were then re-introduced to each LLM with a competition prompt to generate the next round's code submission. Iterative learning in the form of improved final scores was observed for several Round 2 winners, which was based on its prior round competition code, competitors' code, and results. Gpt-5.2-pro and Gemini 2.5 Pro API were effective at iterative learning on the FAERS dataset goal; while Kimi K2 Thinking saw the biggest single round score increase at +0.405. Contestant models were from xAI, OpenAI, Gemini, Claude, DeepSeek, Kimi, GLM, MiniMax, and Qwen manufacturers.

**Keywords:** Goal Oriented Software Engineering · LLM Code Generation · Iterative Learning · FDA FAERS

---

## Table of Contents

1. Introduction
   - 1.1 Human Code Iteration
   - 1.2 AI Code Iteration
   - 1.3 LLM Code Iteration
2. Methods
3. FDA FAERS Data Files
4. Results
   - 4.1 FAERS & Rounds 1-3
   - 4.2 Round 4 Code Results
   - 4.3 Round 4 Notebook Results
   - 4.4 Reference-Based Scoring Metrics
   - 4.5 Round 4 Final Results
5. Discussion
   - 5.1 Academic Paper References
   - 5.2 Difficulty Benchmarking
   - 5.3 Difficulty Benchmarking Standards
6. Limitations and Future Work
7. Conclusions
8. Prompts
9. Data Availability
10. References
11. Acknowledgments
12. Ethical disclosures
13. Rights and permissions
14. Cite this article
