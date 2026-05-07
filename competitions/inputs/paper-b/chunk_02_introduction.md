# Section 1: Introduction

## 1.1 Human Code Iteration

Humans have been iterating on code for many decades, continually evolving their techniques to obtain maximum efficiency [04IntroPythonSpeed, 07IntroNagappan]. What started as labor-intensive writing and debugging in low level languages has evolved into higher-level language automation [01IntroAntoniou]. Additionally, interactive debugging and profiling tools have made code iteration cycles faster. For example, a programmer can run a Python script, profile its performance, and identify bottlenecks in seconds. Similarly, a programmer's choice of top performing algorithms or data structures can yield large performance gains. Humans can effectively leverage their understanding of the problem domain and context, iterating through memory analyzers to reduce bloat. For instance, developers systematically optimized the interpreter in version Python 3.11, yielding an average 1.22× speedup over Python 3.10, with some benchmarks running 10–60% faster [02IntroPython311].

The 2020s introduced structured productivity frameworks like DevOps Research and Assessment's DORA metrics (Deployment frequency, Lead time, Change failure rate, Mean time to restore) [05IntroDORAmetrics], and Microsoft's SPACE framework (Satisfaction, Performance, Activity, Communication, Efficiency) [06IntroSPACE]. However, humans have been found to be limited to code review effectiveness peaks at under 400 lines per session [03Intro300Lines]. Global optimization is extremely challenging for humans to perform; and optimizing an entire large codebase or complex system is often beyond what a single human (or team) can manage.

- Manual optimization is time-consuming and labor-intensive
- Humans tend to stop once "fast enough" performance is reached due to fatigue
- Diminishing returns: A lot of effort for small gains in obtaining last 3% improvement

---

## 1.2 AI Code Iteration

AI code iteration utilizes automated algorithms or machine intelligence to iteratively refine code. This includes search-based optimization, genetic programming, automated program repair, reinforcement learning, and ML-guided compilers [08IntroAIAlgorithms]. DeepMind's 2022 AlphaTensor project applied reinforcement learning to discover new matrix multiplication algorithms, finding math formulas more efficiently and reducing the number of operations needed [10IntroAlphaTensor, 11IntroATNature]. DeepMind's 2023 AlphaDev also utilized reinforcement learning in treating algorithm discovery as a game. AlphaDev found new sorting algorithms that surpassed decades-old human benchmarks, and sorting routines were up to 70% faster for short sequences; while large sequences saw a 1.7% speed improvement in the C++ standard sorting [09IntroAlphaDev, 12IntroADNature]. Bowtie2 by Langdon, W. et al. optimized the DNA sequence alignment program based on a complex 50,000-line bioinformatics tool. By evolving the C++ code, the AI produced a new version that ran on average 70× faster [13IntroLangdon].

Conversely, AI code iterations are rigid, in that the code must be machine-readable, and the improvement goal must be encoded as a fitness function or reward. This lack of flexibility means these systems only accept certain input formats (e.g. a C++ function plus a test suite). AI systems (pre-LLM) do not provide rationales or high-level explanations for the changes to users; they just output the new code. This opacity means developers might be hesitant to trust and deploy such code. In practice, some AI-discovered solutions have to be carefully reviewed and proved equivalent to the original [14IntroQuantNet].

- AI code iteration cannot understanding natural language intent or high-level goals
- Concrete metric or formal specification requires expertise for success at scale
- AlphaDev required framing as sorting assembly instructions with a defined reward [12IntroADNature]

---

## 1.3 LLM Code Iteration

LLM code iteration builds on AI code iteration, but now with mixed language and high contextual awareness capabilities. LLMs have been trained on large portions of internet code and discussions (GitHub repositories [23IntroGitHub], Stack Overflow Q&A [24IntroStack], etc.). DeepMind's 2025 AlphaEvolve used algorithmic AI agents combining evolutionary search with LLM-guided code generation to iteratively evolve algorithms [21IntroAlphaEvolve, 22IntroAlphaEvolvearXiv]; producing a hashing algorithm 30% faster than a widely-used human-designed hash. Contemporary Opus 4.5 is "the coding and agentic workflow leader", scoring 74.4% on a stringent SWE-bench to iteratively plan, code, and refine [17SWEbenchLead].

OpenAI Gpt-5.2's key strength is speed: Gpt-5.2 processes about 187 tokens/second, which is 3.8× faster than Claude which makes iterations faster [19IntroDigApplied]. Gemini 3 Pro's focus on multimodal input at 1 million token context window can iterate faster through entire codebases in a single prompt [Gemini3]. The recent DeepSeek V3.2 matches Western models on many benchmarks and boasts much lower cost to make workflows go farther at $0.56/$1.68 for 1M input and 1M output vs. $5/$25 for Opus [Opus45] and $1.75/$14.00 for Gpt 5.2 [20IntroOpenAICost] and $2/$12 for Gemini 3 [Gemini3].

- LLM interactive prompting and code refinement for generating revised versions
- Human in the loop collaboration for effectively noticing a wide range of issues [16IntroMiniMaxIR]
- Encyclopedic knowledge of optimization tricks, library functions, and idiomatic improvements
