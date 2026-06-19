## ChatGPT 5.5 Thinking Extended 18Jun26

### Prompts
- "list of fda llm approvals"
- "Describe in detail why LLMs are not yet approved."
- "Provide full output with text references in bibtex format at the end. Do not clip any of the existing text."

### Output
"Below is the full unabridged answer, with inline text-reference keys and BibTeX entries at the end.

Why LLMs are not yet FDA-approved

The core reason is not that FDA has decided LLMs can never be medical devices. It is that FDA authorizes specific products for specific intended uses, while most LLMs are broad, open-ended, frequently updated systems whose outputs are difficult to bound, validate, explain, and monitor at the safety level expected for clinical use.

Where FDA is today

FDA already authorizes many AI-enabled medical devices, but those are mostly narrow tools: image analysis, signal analysis, triage, quantification, detection, diagnosis support, planning, and similar bounded functions. FDA says its AI-enabled device list identifies devices authorized for U.S. marketing and that listed devices have met applicable premarket requirements, including review of safety and effectiveness for the device’s intended use and technology. FDA also says the list is not comprehensive and that it is still exploring how to identify and tag devices that incorporate foundation models, including LLMs and multimodal models [@FDA_AI_Devices_2026].  ￼

An independent 2025 review of 1,016 FDA AI/ML device authorizations found that the current landscape is dominated by image-based tools and did not find evidence of LLMs in the studied FDA-authorized device list. The authors note that future LLM devices may fit existing categories if they perform a narrow clinical function, such as estimating sepsis risk from notes, while newer uses like report generation may need new taxonomy categories [@Singh_2025].  ￼

The main blockers

1. FDA does not approve “a model class”; it approves a product with a defined intended use

A general LLM can answer patient questions, summarize notes, suggest diagnoses, draft orders, write letters, explain labs, rank treatments, and more. That breadth is useful commercially, but it is a regulatory problem.

For FDA review, the sponsor needs to say something like: “This device is intended to do X, for Y users, in Z patient population, using A inputs, producing B outputs, under C conditions.” Many LLM products are instead marketed as flexible assistants. That makes the intended use hard to pin down, and FDA’s safety/effectiveness review depends heavily on intended use, patient population, user type, and workflow. FDA’s January 2025 draft guidance for AI-enabled device software focuses on the contents of marketing submissions needed to support FDA’s evaluation of safety and effectiveness across the total product lifecycle [@FDA_AI_Device_Software_2025].  ￼

2. The input space is effectively unbounded

A radiology AI device might take a specific kind of CT image and output a lung nodule flag. A cardiology AI might take a 12-lead ECG and output a specific risk score. Those are hard, but the input/output space is relatively constrained.

An LLM may receive free-text notes, incomplete histories, patient messages, pasted labs, dictated conversations, retrieval snippets, images, and user prompts. Small wording changes can change the answer. Missing context can change the answer. Different institutions document differently. Different patient populations use different language. A validation study has to show performance over the real input distribution, including edge cases, ambiguous cases, and messy clinical documentation. That is much harder than validating a fixed classifier on a defined imaging dataset.

3. LLMs generate fluent errors

LLMs do not merely return a numeric false positive or false negative. They can generate plausible, confident, clinically framed text that is wrong, unsupported, or incomplete. In medical contexts, that matters because a single unsupported statement can affect diagnosis, treatment, consent, documentation, or follow-up.

A 2025 npj Digital Medicine study on LLM clinical text summarization emphasized that fidelity to ground truth is critical to avoid patient-safety risk, and it measured hallucinations and omissions even in a constrained documentation task [@Asgari_2025].  ￼ Another 2025 JMIR study found hallucination rates of 21–50% in LLM conversion of clinical trial eligibility criteria into structured database queries, showing how reliability problems can appear even in semi-administrative clinical research workflows [@Lee_2025].  ￼

The issue is not that every LLM output is unsafe. The issue is that FDA needs evidence that the error rate, error severity, detection mechanisms, user interface, and mitigation strategy are acceptable for the claimed clinical use.

4. Clinical validation is not the same as passing medical exams

Many LLMs perform well on medical question-answering benchmarks. That is encouraging, but FDA would usually need evidence closer to the actual intended use: real users, real workflow, real patient mix, real data quality problems, and clinically meaningful outcomes or performance endpoints.

A model that answers board-style questions well may still fail when facing an incomplete triage note, contradictory chart history, undocumented patient preference, unusual comorbidity, medication reconciliation error, or local protocol. The harder the claim—diagnosis, treatment recommendation, triage, risk prediction—the more robust the validation burden becomes.

5. Transparency and explainability are difficult

FDA and international regulators emphasize transparency for machine-learning medical devices: users should understand intended use, performance, limitations, and, where possible, the basis or logic for outputs. FDA’s transparency principles say transparency is important for safety and effectiveness, can help users evaluate risks and benefits, detect errors or performance decline, and assess bias [@FDA_Transparency_2024].  ￼

LLMs often struggle here. A model may provide an answer without a reliable trace of which facts drove it. Citations can be fabricated unless retrieval and source controls are engineered carefully. Chain-of-thought-style explanations are not necessarily faithful to the model’s true reasoning. For clinical decision support, FDA also distinguishes software that lets clinicians independently review the basis of recommendations from software that effectively asks them to rely on the output. FDA’s 2026 clinical decision support guidance says certain CDS functions can be excluded from the device definition only if they meet statutory criteria, including enabling the health care professional to independently review the basis for the recommendation [@FDA_CDS_2026].  ￼

6. Human factors and automation bias are especially risky with language

LLMs are persuasive. They speak in complete sentences, adapt to the user, and can sound more certain than the evidence warrants. That creates overreliance risk: a clinician, nurse, patient, or call-center worker may trust a well-written answer more than a vague alert or numeric score.

For a regulated LLM device, FDA would likely scrutinize the user interface: Does it show uncertainty? Does it force clinician review? Does it display source evidence? Does it prevent unsupported recommendations? Does it distinguish “draft” from “final”? Does it avoid giving patient-facing instructions that exceed its intended use? These are not just product-design questions; they are safety controls.

7. LLMs change too often

FDA can handle software updates, and it has created a mechanism called a Predetermined Change Control Plan for AI-enabled devices. But the plan has to describe the planned modifications, methodology to develop/validate/implement those modifications, and an impact assessment; FDA reviews the plan as part of the marketing submission [@FDA_PCCP_2025].  ￼

Modern LLM products may change model weights, retrieval corpora, prompts, safety layers, tool integrations, APIs, context windows, embedding models, and vendor infrastructure. A small upstream model update can change downstream clinical behavior. For FDA authorization, a sponsor would need a controlled versioning and change-management strategy. “The vendor updated the model last night” is not compatible with the usual medical-device evidence model unless the update path is pre-specified, validated, and monitored.

8. Bias and generalizability are not solved

LLMs can encode biases from training data, documentation patterns, and medical literature. In healthcare, that can become unequal recommendations, unequal quality of explanations, or missed conditions in underrepresented groups. A Nature Medicine paper on health-equity harms in LLMs argues that LLMs may introduce harm and exacerbate disparities, and that reliable evaluation of equity-related failures is a critical step toward safer systems [@Pfohl_2024].  ￼

For FDA, this maps to subgroup performance, representativeness of validation data, real-world monitoring, and mitigation plans. A model that performs well on average but fails in a subgroup can still be unsafe.

9. Security and data governance are harder with LLM systems

LLM medical products often connect to EHRs, retrieval systems, note databases, patient portals, dictation tools, and external APIs. That raises risks around protected health information, prompt injection, data leakage, corrupted retrieval content, malicious inputs, and model behavior under adversarial prompts.

FDA’s cross-center AI paper identifies transparency, explainability, governance, bias, cybersecurity, quality assurance, robustness, and resilience to changing clinical inputs as key areas for AI in medical products [@FDA_AI_Medical_Products_2025].  ￼ Those issues are magnified when the model is interactive and generative rather than a fixed detector.

What would likely get authorized first

The first LLM-based FDA-authorized products are likely to be narrow, locked-down, clinician-in-the-loop tools, not general “AI doctor” chatbots. Examples that are more plausible:

A tool that drafts a radiology impression from structured findings, but requires radiologist approval.

A tool that extracts specific facts from clinical notes into a structured form, with source highlighting and human verification.

A tool that summarizes a patient encounter for documentation, with explicit omission/hallucination monitoring and a workflow that keeps the clinician responsible for final note signing.

A tool that answers questions only from a locked, validated medical knowledge base and cites the exact source passages.

The successful version would likely need a frozen or tightly controlled model, a precise indication for use, bounded inputs and outputs, prospective or strong retrospective validation, subgroup analyses, human-factors testing, source-grounding, monitoring for drift, cybersecurity controls, and a PCCP for updates.

Bottom line

LLMs are not yet FDA-approved as medical products because they are too general, too variable, too hard to validate exhaustively, too opaque, too update-prone, and too prone to fluent but unsafe errors for broad clinical claims. FDA is not blocking AI broadly; it has authorized many AI-enabled devices. But today’s authorized AI devices are mostly narrow tools with well-defined inputs, outputs, users, and clinical functions. LLMs will likely reach FDA authorization when developers package them as specific, controlled medical-device functions rather than open-ended clinical assistants.

BibTeX references

@misc{FDA_AI_Devices_2026,
  author       = {{U.S. Food and Drug Administration}},
  title        = {Artificial Intelligence-Enabled Medical Devices},
  year         = {2026},
  howpublished = {FDA web page},
  url          = {https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices},
  urldate      = {2026-06-18},
  note         = {AI-enabled medical device list and FDA statements on foundation-model/LLM tagging}
}

@article{Singh_2025,
  author  = {Singh, Rohan and Bapna, Monika and Diab, Abdul Rahman and Ruiz, Emily S. and Lotter, William},
  title   = {How AI is used in FDA-authorized medical devices: a taxonomy across 1,016 authorizations},
  journal = {npj Digital Medicine},
  volume  = {8},
  pages   = {388},
  year    = {2025},
  doi     = {10.1038/s41746-025-01800-1},
  url     = {https://www.nature.com/articles/s41746-025-01800-1}
}

@misc{FDA_AI_Device_Software_2025,
  author       = {{U.S. Food and Drug Administration}},
  title        = {Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations},
  year         = {2025},
  month        = jan,
  howpublished = {Draft guidance for industry and Food and Drug Administration staff},
  url          = {https://www.fda.gov/regulatory-information/search-fda-guidance-documents/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing},
  urldate      = {2026-06-18}
}

@article{Asgari_2025,
  author  = {Asgari, Elham and Monta{\~n}a-Brown, Nina and Dubois, Magda and Khalil, Saleh and Balloch, Jasmine and Au Yeung, Joshua and Pimenta, Dominic},
  title   = {A framework to assess clinical safety and hallucination rates of LLMs for medical text summarisation},
  journal = {npj Digital Medicine},
  volume  = {8},
  pages   = {274},
  year    = {2025},
  doi     = {10.1038/s41746-025-01670-7},
  url     = {https://www.nature.com/articles/s41746-025-01670-7}
}

@article{Lee_2025,
  author  = {Lee, K. H. and Jang, S. and Kim, G. J. and Park, S. and Kim, D. and Kwon, O. J. and Lee, J. H. and Kim, Y. H.},
  title   = {Large Language Models for Automating Clinical Trial Criteria Conversion to Observational Medical Outcomes Partnership Common Data Model Queries: Validation and Evaluation Study},
  journal = {JMIR Medical Informatics},
  volume  = {13},
  pages   = {e71252},
  year    = {2025},
  doi     = {10.2196/71252},
  pmid    = {41100527},
  pmcid   = {PMC12530336},
  url     = {https://medinform.jmir.org/2025/1/e71252}
}

@misc{FDA_Transparency_2024,
  author       = {{U.S. Food and Drug Administration}},
  title        = {Transparency for Machine Learning-Enabled Medical Devices: Guiding Principles},
  year         = {2024},
  month        = jun,
  day          = {13},
  howpublished = {FDA web page},
  url          = {https://www.fda.gov/medical-devices/software-medical-device-samd/transparency-machine-learning-enabled-medical-devices-guiding-principles},
  urldate      = {2026-06-18}
}

@misc{FDA_CDS_2026,
  author       = {{U.S. Food and Drug Administration}},
  title        = {Clinical Decision Support Software: Guidance for Industry and Food and Drug Administration Staff},
  year         = {2026},
  month        = jan,
  howpublished = {Final guidance},
  url          = {https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software},
  urldate      = {2026-06-18}
}

@misc{FDA_PCCP_2025,
  author       = {{U.S. Food and Drug Administration}},
  title        = {Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions},
  year         = {2025},
  month        = aug,
  day          = {18},
  howpublished = {FDA guidance},
  url          = {https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence},
  urldate      = {2026-06-18}
}

@article{Pfohl_2024,
  author  = {Pfohl, Stephen R. and Cole-Lewis, Heather and Sayres, Rory and Neal, Darlene and Asiedu, Mercy and Dieng, Awa and Tomasev, Nenad and Rashid, Qazi Mamunur and Azizi, Shekoofeh and Rostamzadeh, Negar and McCoy, Liam G. and Celi, Leo Anthony and Liu, Yun and Schaekermann, Mike and Walton, Alanna and Parrish, Alicia and Nagpal, Chirag and Singh, Preeti and Dewitt, Akeiylah and Mansfield, Philip and Prakash, Sushant and Heller, Katherine and Karthikesalingam, Alan and Semturs, Christopher and Barral, Joelle and Corrado, Greg and Matias, Yossi and Smith-Loud, Jamila and Horn, Ivor and Singhal, Karan},
  title   = {A toolbox for surfacing health equity harms and biases in large language models},
  journal = {Nature Medicine},
  volume  = {30},
  pages   = {3590--3600},
  year    = {2024},
  doi     = {10.1038/s41591-024-03258-2},
  url     = {https://www.nature.com/articles/s41591-024-03258-2}
}

@misc{FDA_AI_Medical_Products_2025,
  author       = {{U.S. Food and Drug Administration}},
  title        = {Artificial Intelligence and Medical Products: How CBER, CDER, CDRH, and OCP are Working Together},
  year         = {2025},
  month        = feb,
  howpublished = {FDA cross-center paper, revised February 2025},
  url          = {https://www.fda.gov/media/177030/download},
  urldate      = {2026-06-18},
  note         = {Original March 2024 paper revised February 2025}
}"
