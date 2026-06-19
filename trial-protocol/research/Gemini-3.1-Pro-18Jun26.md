## Gemini 3.1 Pro 18Jun26

### Prompts
- "List of fda llm approvals"
- "Describe in detail why LLMs are not yet approved."
- "Provide full output with text references in bibtex format at the end. Do not clip any of the existing text."
- "Retry using fda_2025_dhac and other references exactly in the body text for each time that information was used."

### Output
"As of 2026, there are currently **zero** FDA-approved or cleared Large Language Models (LLMs) or generative AI medical devices.
While the FDA has authorized over 1,450 artificial intelligence and machine learning (AI/ML) medical devices to date, these approvals are entirely for **predictive or narrow AI**—such as algorithms that detect anomalies in radiology scans or analyze ECG rhythms [pmc_2025_illusion_ai]. The agency has not yet granted authorization to any system powered by an LLM or generative AI for use as a clinical decision support system (CDSS) or diagnostic tool.
Here is a detailed breakdown of why LLMs are currently absent from the FDA's approved list, the specific technical, clinical, and legal hurdles preventing them from achieving FDA approval as Software as a Medical Device (SaMD), and how the regulatory landscape is shifting.
### Why the FDA Hasn't Approved LLMs Yet
The traditional regulatory framework used by the U.S. Food and Drug Administration (FDA) was designed decades ago for static hardware and deterministic software [legis1_2026_fda_ai]. To clear a medical device, a developer must prove that the product is safe, effective, and performs predictably within a strictly defined scope. LLMs fundamentally break this traditional paradigm and present several unique regulatory hurdles that the agency is currently trying to solve:
#### 1. The Stochastic Nature of Generative AI vs. Deterministic Software
Traditional medical software and algorithms are **deterministic**: if you feed the exact same patient data into the system, you get the exact same result every time (a 100% predictable output). This makes the software easy to test, validate, and audit.
LLMs are inherently **stochastic** (probabilistic). If a clinician inputs the exact same patient scenario into an LLM ten different times, the model might generate ten differently phrased responses, potentially with varying levels of nuance or clinical emphasis. The FDA currently lacks a standardized methodology to clinically validate software that produces highly variable, unpredictable outputs, making traditional safety and efficacy testing incredibly difficult [legis1_2026_fda_ai].
#### 2. Software of Unknown Provenance (SOUP)
When submitting software for FDA clearance, developers must provide extensive documentation on how the system was built, known as traceability. This includes:
 * Exactly what data the system was trained on.
 * How the data was weighted.
 * End-to-end audit trails of the logic used to arrive at a conclusion.
Commercial LLMs are effectively "black boxes" trained on vast, uncurated scrapes of the internet [pmc_2025_illusion_ai]. They often rely on datasets containing outdated medical guidelines, biased demographic data, and outright misinformation. Because developers cannot easily trace how an LLM arrived at a specific medical conclusion or prove the purity of its training data, they cannot meet the FDA's strict documentation and traceability standards.
#### 3. The "Scope Creep" and Intended Use Problem
To get FDA approval, a device must have a highly specific **Intended Use** statement (e.g., "This software is intended to highlight suspected pulmonary nodules on CT scans of adults over 18"). The device is then rigorously tested *only* for that specific use case.
LLMs, however, are generalized reasoners. If an AI is approved to diagnose a specific type of pneumonia, it should only do that. However, researchers have found that it is incredibly difficult to constrain LLMs [pmc_2025_illusion_ai]. Even with strict system prompts, a developer attempting to "fine-tune" an LLM strictly for cardiology triage will find it difficult to prevent the model from answering questions outside its approved scope. A user could easily coax the cardiology LLM into providing unauthorized, "off-label" medical advice on a pediatric dermatology issue, or acting as an unapproved clinical decision support tool for scenarios entirely outside their intended use. This inability to reliably constrain an LLM creates massive liability and regulatory risk, as the device is effectively capable of infinite "off-label" uses.
#### 4. Hallucinations and Plausible Misinformation
The most well-known flaw of LLMs is their tendency to "hallucinate"—generating false information with a high degree of confidence [pmc_2025_illusion_ai]. While an AI misidentifying a pixel on a photograph is relatively harmless if caught by a doctor, a medical LLM fabricating a non-existent drug interaction or suggesting an incorrect dosage in a time-critical emergency presents an unacceptable risk to patient safety. Regulators mandate the elimination of risk to the maximum extent possible, and there is currently no foolproof software architecture that completely eliminates LLM hallucinations.
#### 5. Privacy, Cybersecurity, Model Drift, and Continuous Learning
Medical devices must integrate seamlessly and securely into clinical workflows while maintaining strict adherence to patient privacy laws like HIPAA.
 * **Data Leakage:** Many LLMs send prompts (which may contain Protected Health Information) to external servers for processing, raising massive data privacy concerns.
 * **Model Drift and Continuous Learning:** Generative AI models are capable of adapting and evolving over time. If an LLM continually updates or alters its weights based on new user interactions or after ingesting new data (model drift), it violates the traditional FDA requirement that a cleared device remain exactly in the state it was in when approved [legis1_2026_fda_ai]. A model that passes safety checks today might perform differently six months from now, which traditionally requires a completely new regulatory submission.
### Traditional AI vs. LLMs in Healthcare
| Feature | Traditional AI/ML (1,450+ Approved) | Generative AI / LLMs (0 Approved) |
|---|---|---|
| **Output Nature** | Deterministic (predictable, consistent) | Stochastic (variable, highly dynamic) |
| **Primary Use Cases** | Radiology (75%+), Cardiology, Neurology | Drafting notes, summarizing records, chatbots |
| **Clinical Scope** | Extremely narrow and locked to one task | Broad, generalized, and conversational |
| **FDA Status** | Routinely cleared via the 510(k) pathway [pmc_2025_illusion_ai] | Currently unapproved for clinical decision making |
### The Regulatory Road Ahead
To bridge this gap, the FDA is actively working to close this regulatory blind spot. Recognizing that patients and clinicians are already starting to use commercial LLMs for health queries, the agency is developing new frameworks specifically tailored to generative AI:
 * **Mental Health Focus:** In late 2025, the FDA convened a first-of-its-kind Digital Health Advisory Committee meeting specifically to address "Generative Artificial Intelligence-Enabled Digital Mental Health Medical Devices" [fda_2025_dhac]. They are actively exploring how to safely regulate GenAI therapy and triage chatbots [sidley_2025_genai_mental_health].
 * **Predetermined Change Control Plans (PCCPs):** To deal with models that evolve, the FDA is exploring PCCPs [legis1_2026_fda_ai]. This framework would allow companies to pre-authorize certain types of post-market updates to their LLMs without having to submit a brand new FDA application every time the model learns something new.
 * **New Authorization Pathways & Sandboxes:** Industry experts and the FDA are discussing the creation of entirely new authorization pathways not anchored to specific, single indications, which would allow for the "generalized" decision support that LLMs excel at [sidley_2025_genai_mental_health]. This includes regulatory "sandboxes" to safely test how these tools perform in real-world clinical environments with heavy human-in-the-loop oversight. However, until developers can mathematically guarantee constraints, trace training data, and eliminate hallucinations, autonomous LLMs will remain unapproved for direct clinical decision-making.
### References
```bibtex
@misc{fda_2025_dhac,
  author = {{U.S. Food and Drug Administration}},
  title = {24 Hour Summary of the Digital Health Advisory Committee - November 6, 2025},
  year = {2025},
  url = {https://www.fda.gov/media/189618/download}
}

@article{legis1_2026_fda_ai,
  author = {Legis1},
  title = {FDA Struggles With Regulating Adaptive AI Devices},
  year = {2026},
  url = {https://legis1.com/news/fda-ai-device-regulation-fdas-static-framework}
}

@article{pmc_2025_illusion_ai,
  author = {National Center for Biotechnology Information},
  title = {The illusion of safety: A report to the FDA on AI healthcare product approvals},
  journal = {PMC},
  year = {2025},
  url = {https://pmc.ncbi.nlm.nih.gov/articles/PMC12140231/}
}

@misc{sidley_2025_genai_mental_health,
  author = {{Sidley Austin LLP}},
  title = {U.S. FDA and CMS Actions on Generative AI-Enabled Mental Health Devices Yield Insights Across AI Product Development},
  year = {2025},
  url = {https://www.sidley.com/en/insights/newsupdates/2025/11/us-fda-and-cms-actions-on-generative-ai-enabled-mental-health-devices-yield-insights-across-ai}
}

```"
