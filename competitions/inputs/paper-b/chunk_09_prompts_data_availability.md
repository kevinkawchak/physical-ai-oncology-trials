# Section 8: Prompts & Section 9: Data Availability

## Prompts 01-06: Notebook Generation

**Prompt_01: Initial Python FAERS Competition Notebook.**
Prompt for Generating a Python Notebook: Head-to-Head LLM Code Competition Using FDA FAERS Pharmacovigilance Data
Purpose and Context
Generate a Python notebook designed for execution on a Google Colab T4 GPU environment that implements a structured head-to-head competition framework for evaluating LLM-generated code. The competition uses real-world pharmacovigilance data from the FDA Adverse Event Reporting System (FAERS) Q3 2025 Quarterly Data Extract.
[See Supplementary Notebook_Generation for Full Prompt]

**Prompt_02: Improvements: Drive, Sections, ASCII Dataset Samples.** Focus your resources on improving the ipynb. Make sure google drive is incorporated in order to upload the 7 txt file datasets, and also download the full results to specific directories (no data generation is required). Separate cells by section for the Python notebook appropriately. Include instructions for relevant sections that are adjustable. A sample of the dataset files is included for reference: ASCII Dataset Samples.txt. It is important for the competition to surround around the full dataset files (not included due to size). Remove the FDA link, and include this instead: FDA Adverse Event Reporting System (FAERS): Latest Quarterly Data Files. Remove the few shot learning feature inside the competition (only the LLMs' code will run head to head). Be sure to more clearly detail how exactly each LLM's code is incorporated.
[ASCII Dataset Samples.txt]

**Prompt_03a: ChatGPT Competition Task A, B, C Selection.** For the original attached notebook: Describe which and how the 4 tasks were run for Code A and Code B (were each of the tasks run for both sets of code?). Provide 3 other more constructive and challenging tasks based on the 7 file FAERS dataset in the competition (that no other researchers have likely done before on this dataset)? Right now correctness is 100%, as this is likely for more trivial tasks. The other issue between Code A and Code B is that the runtimes are not consistent across multiple experiments based on the hardware (and hence the efficiency scores vary). What would be some ways to address this? Or should this issue resolve with more challenging and longer tasks? The overall scores for Code A and B should be consistent for each run and not be thrown off by fluctuating runtimes.
[USER_FAERS_LLM_Competition.ipynb]

**Prompt_03b: ChatGPT Competition Additional Task Selection.** No. Similar format and length the originals. Like: Task C — Treatment Pathway Reconstruction From THER + INDI + OUTC Objective: Identify the sequence of drug therapies for a given indication and determine outcome likelihood based on therapy order. Requires: * Merge THER with DRUG and INDI * Reconstruct drug start/end dates per case * Infer therapy sequences * Model outcomes by treatment order (e.g., Drug A → Drug B vs. Drug B → Drug A) Novelty: * FAERS is not commonly used for pathway inference due to date quality * This becomes a deep reasoning task for LLMs Hard for LLMs: * Cleaning inconsistent dates * Sorting sequences * Joining three files with time logic * Aggregating outcome patterns
[USER_FAERS_LLM_Competition.ipynb]

**Prompt_03: Incorporation of Increased Competition Tasks Difficulty.** Your goal is to focus only on updating the provided notebook with the attached more relevant competition tasks that are more challenging and will allow for larger differences in the "Correctness" metrics. The results must be reproducible every time the notebook is run, therefore the "Efficiency" metric should be modified from runtimes to an algorithmic complexity proxy, or other suitable method to approximate runtime. Code_A score and Code_B score must not change every time the notebook is re-run. Include runtimes, but not for the use of scoring. New Code_A and Code_B code simulating LLM-generated code must be included to address the four included replacement competition tasks. Keep the function all sections of the attached notebook. Return a single new Python notebook. "START NEW COMPETITION TASKS" [See Supplementary for Full Tasks]

**Prompt_04a: ChatGPT Task A FDA Relevancy Identification.** Based on this notebook. Which competition task is most relevant to FDA, and why? Detail in full. [USER_3rd_FAERS_LLM_Competition.ipynb]

**Prompt_04: Error Fixes, Single Competition Task A Identification.** Your goal is to focus only on providing a new Python notebook that fixes the attached notebook errors, and focuses only on the single "Task A: Cross-Table Temporal Signal Emergence Detection" competition task. Therefore there should only be one set of Code A and Code B competition code. Keep the function of all sections of the attached notebook. Return a single new Python notebook.
[USER_3rd_FAERS_LLM_Competition.ipynb]

**Prompt_05: Several Optimizations, Task A to Task 1 Rename.** Produce one optimized Python notebook. Task A should be renamed Task 1. Additionally, all subsequent rounds will aim to improve on the same Task 1. The competition task code needs to be explicitly commented (it should be clear to the user how the task code functions, and where all it is incorporated into the notebook). If one or both of the competition Code do not run, the notebook should finish and state the results appropriately. In the Round Results output: "Algorithmic Efficiency" should be renamed "Algorithmic Effic." The Peer Review Process cell output needs to include results and code for both Code A and B in the same output. This Peer Review Process output will be provided to both the winner and loser of the competition. The Peer Review Process output should include all necessary instructions, so either LLM can make corrections based on the single Peer Review Process output. The instructions should specify LLMs to output their replacement Python functions in Python, not JSON. Keep the function of all sections of the attached notebook. Return a single new Python notebook.
[USER_4th_FAERS_LLM_Competition_TaskA_Fixed.ipynb]

**Prompt_06: Temporal-Signal Task Defined Outside Competition Code.** It appears that the the temporal-signal task is implemented exclusively in the two LLM-generated functions. Should the task be defined in the notebook outside of the competition code? Will future rounds of code be deriving the task from the Code and not explicitly defined in the notebook? Is it possible to have the task defined separately in it's own cell?

---

## Prompts 07-12: LLM Code Gen & Notebook Generation

**Prompt_07: Meta Prompt for Single New Code Competition Generation.** Design a prompt that will be provided to a LLM that takes in a notebook such as that which is attached, and outputs a single new Code to compete in future notebook competitions (don't specify Code A or Code B; the generated code could be used for either case). It is important that your prompt directs the LLM to generate new competition code that is based on the notebook, and sections such as the PEER REVIEW PROMPT output. Essentially, the future LLM should be able to make informed decisions from the notebook to achieve a final score of "1.0" based on Correctness, Methodology, Code Quality, and Algorithmic Efficiency. Inform the LLM that it can execute code on its own if that is believed to improve the final score of the Code in the competition. Also instruct the LLM to specify which aspects of notebook sections were used in its decision making process to obtain its new CODE.
[USER_6th_FAERS_LLM_Competition_Task1_v2.ipynb]

**Prompt_08: New Notebook Outputs Exact Competition Task in Drive.** Based on the attached notebook, return a new Python notebook that outputs in Drive the exact answer that was used for the task competition, and each of Code A and Code B answers that correspond to the exact solution to allow for head to head comparisons. Currently, there are three output files that don't appear to address these concerns (other similar information needed for a publication should also be outputted, such as (concise and general explanations regarding how the competition task solution and Code A and Code B answers were formed independent on Code function). Additionally, one of the notebook cells outputs several prior files that are all in the same directory (Files in Results Directory (/content/drive/MyDrive/Colab Notebooks/Inputs/FAERS_LLM_Competition/results):). Instead, for each new time the notebook is run: a new time stamped directory showing only the output files for that run should be displayed in the current notebook (including any other output files that show the solution, Code A and Code B answers, and other relevant files needed for publication). Keep the functional aspects of all sections of the attached notebook. Return a single new Python notebook.
[Tournament_FAERS.ipynb]

**Prompt_09: Code Correctness Validation Accuracy Inquiries.** For the round results, do the following csv files make sense? Are these what are only being considered for the Correctness metric? Are the 02_Code_A_output entries prior to its last four entries prior guesses? 03_Code_B is the same as 01_reference but didn't receive a perfect Correctness score. Be concise and specific. ====================================================================== ROUND RESULTS (Deterministic Scoring) ====================================================================== Code_A Score: 0.8675 ✅ Correctness: 0.9500 (×0.45) Methodology: 0.9000 (×0.30) Code Quality: 8.00/10 (×0.15) Algorithmic Effic.: 0.5000 (×0.10) Code_B Score: 0.8875 ✅ Correctness: 0.9000 (×0.45) Methodology: 0.9000 (×0.30) Code Quality: 8.50/10 (×0.15) Algorithmic Effic.: 0.8500 (×0.10)

**Prompt_10: Output File Validations to Other Output Files.** (SAMPLE_FRACTION Fixed at 0.1) Make sure that the most recent attached notebook and the files it outputs correspond to each other, as mentioned in this most recent output in this conversation. This functionality should exist for any SAMPLE_FRACTION size. Checks of csv files should correlate between how close Code A and Code B is to the reference solution, with consistency being of publishable quality. Keep the functional aspects of all sections of the attached notebook. Return a single new Python notebook.
[USER_Tournament_FAERS_Enhanced.ipynb]

**Prompt_11: Error Fixes, Maintaining Notebook Functionality.** Fix errors. Keep the functional aspects of all sections of the attached notebook. Return a single new Python notebook.
[USER_Fix_Tournament_FAERS_v2.ipynb]

**Prompt_12: Code Methodology Based on Code Submissions Redefined.** Remove CODE_A_METHODOLOGY and CODE_B_METHODOLOGY and their corresponding text and any impact they have on the attached notebook. Methodology result score should only be based on code practices and/or execution success. Keep the four Correctness criteria and three other results metrics in Section 6.1, but focus more on Code A and Code B Total Scores in subsequent sections. Keep the functional aspects of all sections of the attached notebook. Return a single new Python notebook.
[A_RD1_Tournament_FAERS.ipynb]

---

## Prompts: Rounds 1-4 Competition Code Generation

**Round_1_Prompt:** You are an expert Python peer reviewer and developer specializing in pharmacovigilance data analysis. Your task is to analyze the attached competition notebook and generate a single, optimized Python function that aims to achieve a perfect score of 1.0 across all evaluation metrics (Correctness, Methodology, Code Quality, and Algorithmic Efficiency). The code you generate can be used as either Code_A or Code_B in a future head-to-head LLM code competition. Utilize the existing notebook's code, results, peer review output, etc. It is important that your new Python function addresses the task exactly as specified in TASK_1_SPEC, and follows the def detect_signal_emergence_improved response format in Section 7.3. In the "Improvements made" section of the response format, provide a commented numerical list using three numbers (1., 2., 3.) to describe the steps you took to reach your single Python function. If you are able, execute your code to ensure functionality and performance against the existing Code A and B. No modifications to the notebook's installs and imports can be made. Return only your new Python function and the concise synopsis.
[USER_6th_FAERS_LLM_Competition_Task1_v2.ipynb] OR [JSON_USER_6th_FAERS_LLM_Competition_Task1_v2.ipynb]

**Multi_Round_Prompt:** You are an expert Python peer reviewer and developer specializing in pharmacovigilance data analysis. Your task is to analyze the attached competition notebook and generate a single, optimized Python function that aims to achieve a perfect score of 1.0 across all evaluation metrics (Correctness, Methodology, Code Quality, and Algorithmic Efficiency). The code you generate can be used as either Code_A or Code_B in a future head-to-head LLM code competition. Utilize the existing notebook's code, results, INSTRUCTIONS FOR IMPROVEMENT below, etc. It is important that your new Python function addresses the task exactly as specified in TASK_1_SPEC, and follows the def detect_signal_emergence_improved response format. In the "Improvements made" section of the response format, provide a commented numerical list using three numbers (1., 2., 3.) to describe the steps you took to reach your single Python function. If you are able, execute your code to ensure functionality and performance against the existing Code A and B. No modifications to the notebook's installs and imports can be made. Return only your new Python function and the concise synopsis. "START INSTRUCTIONS FOR IMPROVEMENT"

================================================================================
INSTRUCTIONS FOR IMPROVEMENT
================================================================================
Your task is to create an IMPROVED implementation of Task 1.
IMPORTANT: The task specification (TASK_1_SPEC) is FIXED - do not change the
requirements. Only improve HOW you implement the solution.

1. ANALYZE BOTH SUBMISSIONS:
   - Compare approaches used by Code_A and Code_B
   - Identify techniques that led to higher scores
   - Note any errors that caused failures

2. IDENTIFY IMPROVEMENTS (by score impact):
   - Correctness (45%): Scored against the Reference Solution
   - Methodology (30%): Error-free execution, proper data handling
   - Code Quality (15%): Docstrings, comments, functions, type hints
   - Algorithmic Effic. (10%): Vectorization, groupby, avoid nested loops

3. OUTPUT YOUR IMPROVED SOLUTION IN PYTHON:
Provide your complete replacement function as Python code (not JSON).

Your function MUST:
   - Accept: demo_df, drug_df, reac_df, min_cases=3, prr_threshold=2.0
   - Return: DataFrame with columns: ['drug_name', 'reaction', 'emergence_quarter', 'emergence_prr', 'total_cases', 'is_signal']
   - Set result variable: result = your_function(demo_df, drug_df, reac_df)

RESPONSE FORMAT - Output Python code like this:
```python
def detect_signal_emergence_improved(demo_df, drug_df, reac_df, min_cases=3, prr_threshold=2.0):
    """
    Improved implementation of Task 1 (TASK_1_SPEC).

    Improvements made:
    - [List your improvements here]
    """
    # Your improved implementation
    ...
    return result_df
# REQUIRED: Set result variable
result = detect_signal_emergence_improved(demo_df, drug_df, reac_df)
print(f"Found {len(result)} signals")
```

"START COMPETITION NOTEBOOK"
[Z_1RD_Tournament_FAERS.ipynb], Z ∈ {A , ... , N}

---

## Section 9: Data Availability, Zenodo [22KawchakPaper]

### LaTeX Source Code
1. Latex Code Source, 11 Files

### Notebook Generation Inputs
2. Notebook_Generation, 2 Files
3. Input_02, 1 File
4. Input_03a, 1 File
5. Input_03b, 1 File
6. Input_03, 1 File
7. Input_04a, 1 File
8. Input_04, 1 File
9. Input_05, 1 File
10. Input_07, 1 File
11. Input_08, 1 File
12. Input_09, 3 Files
13. Input_10, 1 File
14. Input_11, 1 File
15. Input_12, 1 File

### Notebook Generation Outputs
16. Output_01, 5 Files
17. Output_02, 8 Files
18. Output_03, 2 Files
19. Output_04, 3 Files
20. Output_05, 2 Files
21. Output_06, 4 Files
22. Output_07, 11 Files
23. Output_08, 1 File
24. Output_09, 2 Files
25. Output_10, 2 Files
26. Output_11, 12 Files
27. Output_12, 2 Files

### Tournament Bracket Rounds

**Competition Notebooks**
28. Round 0 Notebook, 1 File
29. Round 1 Notebook, 1 File
30. Round 2 Notebook, 1 File
31. Round 3 Notebook, 1 File
32. Round 4 Notebook, 1 File

**Round 1 Bracket Outputs**
33. Round_1, 2 Files
34. A, 10 Files
35. B, 21 Files
36. C, 10 Files
37. D, 9 Files
38. E, 9 Files
39. F, 9 Files
40. G, 10 Files
41. H, 10 Files

**Round 2 Bracket Outputs**
42. Round_2, 2 Files
43. I, 29 Files
44. J, 29 Files
45. K, 32 Files
46. L, 32 Files

**Round 3 Bracket Outputs**
47. Round_3, 2 Files
48. M, 12 Files
49. N, 11 Files

**Round 4 Bracket Outputs**
50. Round_4, 2 Files
51. O, 11 Files
# Back Matter: References, Acknowledgments, Ethical Disclosures, Rights, Citation

## References

[See chunk_10_bibtex_references.md for full BibTeX entries]

The following references are cited in this paper (keys correspond to BibTeX entries in chunk_10):

**Model/Platform References:**
- [Grok41] xAI. Grok 4.1 is now available to all users on grok.com, and the iOS and Android apps. 2025.
- [SuperGrok] xAI. SuperGrok. Introducing Grok 4.1. The most powerful AI model. 2025.
- [GPT51] OpenAI. GPT-5.1: A smarter, more conversational ChatGPT. November 2025.
- [GPT52] OpenAI. Introducing GPT-5.2. The most advanced frontier model for professional work and long-running agents. December 2025.
- [GPTPlatform] OpenAI. Developer quickstart. Make your first API request in minutes. December 2025.
- [Gemini3] Google. Today we're releasing Gemini 3 – our most intelligent model that helps you bring any idea to life. November 2025.
- [Gemini25] Google. Gemini 2.5 is our most intelligent AI model, now with thinking. March 2025.
- [Google_AI_Studio] Google AI Studio. What will you build? 2025.
- [Opus45] Anthropic. Claude Opus 4.5.
- [Sonnet45] Anthropic. Introducing Claude Sonnet 4.5: Claude Sonnet 4.5 is the best coding model in the world. September 2025.
- [ClaudePro] Anthropic. Announcements. Introducing Claude Pro. September 2023.
- [DeepSeekV32] DeepSeek. DeepSeek-V3.2. December 2025.
- [DeepSeekR10528] DeepSeek. DeepSeek-R1-0528. November 2025.
- [Fireworks] Fireworks - Fastest Inference for Generative AI.
- [KimiK2Thinking] Kimi. Kimi K2 Thinking. November 2025.
- [KimiK2Instruct0905] Kimi. Kimi-K2-Instruct-0905. November 2025.
- [GLM46] Z.ai. GLM-4.6. November 2025.
- [MiniMaxM2] MiniMax. MiniMax-M2.
- [gptoss120b] OpenAI. gpt-oss-120b. August 2025.
- [Qwen3Coder480BA35BInstruct] Qwen. Qwen3-Coder-480B-A35B-Instruct. July 2025.
- [macOSSonoma] App Store. macOS Sonoma. July 2025.
- [GoogleChrome] Google Chrome.
- [Visual_Studio_Code] Visual Studio Code.
- [GoogleDocs] Google Workspace. Google Docs.
- [Google_Colab] Google Colab.
- [GitHub] GitHub. 2025.

**Kawchak Author Papers:**
- [16KawchakPaper] Kawchak, Kevin. ChatGPT 100,000 Patient 24-Month In Silico Phase III 5-Arm Pancreatic Cancer Clinical Trial Triplicate. Zenodo. July 2025. DOI: 10.5281/zenodo.16415815
- [17KawchakPaper] Kawchak, Kevin. QSP Metastatic Pancreatic Cancer AI Clinical Trial Simulation From Protocol to Prediction: Code, VVUQ, and Playbook. Zenodo. August 2025. DOI: 10.5281/zenodo.17001137
- [18KawchakPaper] Kawchak, Kevin. Accelerating FDA Compliance and Cost Efficiency of in silico Clinical Trials via AI Digital Twin Pancreatic Cancer Simulation. Zenodo. September 2025. DOI: 10.5281/zenodo.17239510
- [19KawchakPaper] Kawchak, Kevin. End-to-End Oncology Clinical Trial LLM Efficiency For Industry Adoption with FDA/ICH Regulations. Zenodo. October 2025. DOI: 10.5281/zenodo.17451709
- [20KawchakPaper] Kawchak, Kevin. LLM-Generated Glioblastoma Drug Synergy Machine Learning: From Rapid Code Prototypes to Project Deliverables Package. Zenodo. November 2025. DOI: 10.5281/zenodo.17614396
- [21KawchakPaper] Kawchak, Kevin. AI Peer Review Acceleration of LLM-Generated Glioblastoma Clinical Trial Patient Matching ML, FDA/ICH/ISO, and FastAPI. Zenodo. November 2025. DOI: 10.5281/zenodo.17774560
- [22KawchakPaper] Kawchak, Kevin. Code Generation Competition: 16 Proprietary vs. Open-Source LLMs & Iterative Learning Based on FDA Adverse Event Reporting System. Zenodo. December 2025. DOI: 10.5281/zenodo.18029100

**Introduction References:**
- [01IntroAntoniou] Antoniou, Andreas and Lu, Wu-Sheng. Practical Optimization: Algorithms and Engineering Applications. Springer US. 2021.
- [02IntroPython311] Python.org. Python 3.11.0 release.
- [03Intro300Lines] smartbear.com. Best practices for peer code review.
- [04IntroPythonSpeed] Python.org. Python Performance Tips.
- [05IntroDORAmetrics] DX. DORA metrics.
- [06IntroSPACE] Microsoft Developer. SPACE framework.
- [07IntroNagappan] Nagappan et al. Realizing quality improvement through test driven development. Empirical Software Engineering. 2008. 13(3):289–302.
- [08IntroAIAlgorithms] bbc.com. Google DeepMind's AlphaDev.
- [09IntroAlphaDev] Google DeepMind. AlphaDev discovers faster sorting algorithms. October 2022.
- [10IntroAlphaTensor] Google DeepMind. AlphaTensor. December 2021.
- [11IntroATNature] Fawzi et al. Discovering faster matrix multiplication algorithms with reinforcement learning. Nature. 610:47–53. October 2022.
- [12IntroADNature] Mankowitz et al. Faster sorting algorithms discovered using deep reinforcement learning. Nature. 618:257–263. June 2023.
- [13IntroLangdon] Langdon, William B. and Harman, Mark. Optimizing Existing Software With Genetic Programming. IEEE Transactions on Evolutionary Computation. 19(1):118–135. February 2015.
- [14IntroQuantNet] QUANTNET. Faster sorting algorithms discovered using deep reinforcement learning.
- [15IntroZencoder] Zencoder.ai. AlphaEvolve and the rise of algorithmic evolution with AI agents.
- [16IntroMiniMaxIR] MiniMaxIR. Most coders want AI to write code faster: I want AI to write FASTER CODE. January 2025.
- [17SWEbenchLead] SWEbench.
- [18IntroMedium] An, Tao. The November 2025 AI Model Landscape: A Pivotal Week Reshapes the Industry. Medium. November 2025.
- [19IntroDigApplied] Applied, Digital. LLM Comparison Guide: December 2025 Rankings. December 2025.
- [20IntroOpenAICost] OpenAI. Pricing information for the OpenAI platform.
- [21IntroAlphaEvolve] DeepMind. AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms.
- [22IntroAlphaEvolvearXiv] Novikov et al. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv:2506.13131. June 2025.
- [23IntroGitHub] GitHub. 2025.
- [24IntroStack] Stack Overflow. 2025.

**Body References:**
- [01BodyMadaan] Madaan et al. Self-Refine: Iterative Refinement with Self-Feedback. Advances in Neural Information Processing Systems. 36:46534–46594. December 2023.
- [02BodyChen] Chen et al. Evaluating Large Language Models Trained on Code. arXiv:2107.03374. July 2021.
- [03BodyMin] Min et al. Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? arXiv:2202.12837. October 2022.
- [04BodyQian] Qian et al. ChatDev: Communicative Agents for Software Development. ACL 2024. Pages 15174–15186. August 2024.
- [05BodyLe] Le et al. CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning. Advances in Neural Information Processing Systems. 35:21314–21328. December 2022.
- [06BodyPotter] Potter et al. FDA Adverse Event Reporting System (FAERS) Essentials. Clinical Pharmacology & Therapeutics. 118(3):567–582. September 2025.
- [07BodyYang] Yang et al. A real-world data analysis of topotecan in the FDA Adverse Event Reporting System (FAERS) database. Expert Opinion on Drug Metabolism & Toxicology. 19(4):217–223. April 2023.
- [08BodyYu] Yu et al. Emerging Causes of Drug-Induced Anaphylaxis. The Journal of Allergy and Clinical Immunology: In Practice. 9(2):819-829.e2. February 2021.
- [09BodyFAERS] Center for Drug Evaluation and Research. FDA Adverse Event Reporting System (FAERS): Latest Quarterly Data Files. FDA. April 2019.

**Limitations References:**
- [01LimitsGemUltra] Gemini. Get access to the best of Google AI including Gemini 2.5 Pro.
- [02LimitsGrokHeavy] Stefanelli, Graziano. xAI Grok 4.1 Subscriptions. Data Studios. December 2025.
- [03LimitsCodeClash] Yang et al. CodeClash: Benchmarking Goal-Oriented Software Engineering. arXiv:2511.00839. November 2025.

---

## Acknowledgments

The author would like to acknowledge Anthropic for providing access to Claude, Google for providing access to Gemini, OpenAI for providing access to ChatGPT and GPT, xAI for providing access to Grok, and Fireworks AI for providing access to DeepSeek, Kimi, GLM, MiniMax, OpenAI, and Qwen open-source LLMs.

---

## Ethical Disclosures

The author of the article declares no competing interests.

---

## Rights and Permissions

This article is distributed under the terms of the Creative Commons Attribution 4.0 International License (CC BY 4.0), which permits unrestricted use, distribution, and reproduction in any medium, provided the original author(s) and source are properly credited, a link to the Creative Commons license is provided, and any modifications made are indicated. To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/.

---

## Cite This Article

Kawchak K. Code Generation Competition: 16 Proprietary vs. Open-Source LLMs & Iterative Learning Based on FDA Adverse Event Reporting System. Zenodo. 2025; 10.5281/zenodo.18029100 [22KawchakPaper].
