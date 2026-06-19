## Gemini-3.1-Pro-19Jun26

### Prompts
- Only provide clinical trial options. Provide the number of times each were used in the last 5 years for cancer treatments. Further detail exactly what each is, and how to state each in headlines and literature. Expanded Access (Compassionate Use): This is the most common pathway. It is an FDA-regulated process that allows patients with immediately life-threatening conditions or serious diseases to gain access to an investigational medical product (drug, biologic, or medical device) outside of a clinical trial. The patient's doctor must apply to the FDA and the drug manufacturer, arguing that the potential patient benefit justifies the potential risks of the treatment.
Emergency IND (Investigational New Drug): A specific type of Expanded Access used in urgent, life-threatening situations where there isn't enough time to go through the standard Compassionate Use review process. The FDA can grant authorization over the phone.
- Provide the exact trial type and information needed for the following.

### Output
Exact classification of the uploaded proposal

The LaTeX source does not yet define an exact trial type. It repeatedly uses “Physical AI oncology clinical investigation,” while:
	•	sections/s4-comparative.tex classifies the robot-control software as a medical device.
	•	Proposed §515D(j) in sections/s3-amendment.tex preserves the investigational device exemption, or IDE, framework.
	•	Proposed §515D(i) does not define “Physical AI oncology clinical investigation.”
	•	The reference to the IND framework is relevant only when an investigational drug or biologic is also being studied; it does not make the robotic-device study an IND trial.

The correct umbrella classification is:

Prospective interventional medical-device clinical investigation conducted under 21 CFR part 812.

It should not ordinarily be described as a Phase I, II, or III trial. ClinicalTrials.gov uses Phase: Not Applicable for device trials.  ￼

Exact initial trial type

For the first patient-facing investigation contemplated by the document, the most defensible designation is:

A prospective, interventional, open-label, single-group, first-in-human early feasibility study of a significant-risk investigational medical device conducted under an FDA-approved IDE; Phase: Not Applicable; Primary Purpose: Device Feasibility.

FDA defines an early feasibility study as a limited study conducted early in device development, usually with a small number of participants, to evaluate initial clinical safety, device functionality, and proof of principle and potentially guide device modifications. Early feasibility studies of significant-risk devices require an IDE.  ￼

Risk designation

Based on the source’s patient-contact robotic motion, surgical and needle-placement functions, force control, vascular “no-fly” zones, emergency stops, and possible catastrophic failures, significant-risk device investigation is the appropriate presumptive classification for the surgical, therapeutic-positioning, and needle-placement applications.

That conclusion is an informed regulatory inference rather than a binding FDA determination. A significant-risk device includes one used for a substantially important diagnostic or therapeutic function that presents a potential for serious risk. The reviewing IRB and FDA ultimately determine whether the investigation is significant risk.  ￼

Exact ClinicalTrials.gov entries

Registration field	Recommended entry
Study Type	Interventional
Study Phase	Not Applicable
Intervention Type	Device
Primary Purpose	Device Feasibility
Intervention Model	Single Group Assignment
Allocation	N/A
Masking	None—Open Label
Masking enhancement	Masked independent outcomes assessor, when feasible
Number of Arms	1
Arm Type	Experimental
FDA-Regulated Device Product	Yes
U.S. FDA IDE	Yes
First-in-Human	Yes
Official Title	A Prospective, First-in-Human Early Feasibility Study of the [Device Name] Physical AI Robotic System for [Procedure] in Adults With [Cancer Type]
Brief Title	Early Feasibility Study of [Device] in [Cancer Type]
Condition	Exact cancer diagnosis and stage—not merely “oncology”
Intervention Name	Full device name, model, software version, and autonomy configuration
Enrollment	Small, justified by the safety and feasibility objectives
Study Sites	Each participating institution and principal investigator
Follow-up	Procedure-specific acute and longer-term safety follow-up

ClinicalTrials.gov defines “Device Feasibility” as a small trial—generally fewer than 10 participants—evaluating feasibility or a prototype rather than health outcomes. “Not Applicable” is the correct phase entry for trials without FDA-defined drug phases.  ￼

A sequential model may replace single-group assignment when the protocol uses staged enrollment, such as:
	1.	Surgeon-controlled robotic assistance.
	2.	Supervised semiautonomous operation.
	3.	Higher-autonomy operation only after a safety review.

Information that must be specified

1. Device identity

The protocol and bill record must provide:
	•	Device trade and generic name.
	•	Manufacturer and legal sponsor.
	•	Hardware model and configuration.
	•	Software, firmware, model, and algorithm versions.
	•	Unique device or configuration identifier.
	•	Intended use.
	•	Exact oncology indication.
	•	Exact procedure performed.
	•	Patient-contact components.
	•	Whether the device is reusable, disposable, implanted, or procedure-limited.
	•	Proposed 510(k), De Novo, PMA, or HDE marketing pathway.

An IDE application must contain the device’s intended use, description, principles of operation, manufacturing controls, labeling, prior testing, and investigational plan.  ￼

2. Investigational status

State explicitly:

The [device name] is an investigational medical device. It has not been cleared or approved by FDA for the investigational use evaluated in this study.

Also document:
	•	Significant-risk determination.
	•	IDE application number once assigned.
	•	FDA IDE status.
	•	IRB approval status for every site.
	•	Prior Q-Submission or Pre-Submission number.
	•	Whether any component is already cleared but is being studied for a new intended use.
	•	Whether an investigational drug, biologic, imaging agent, or combination product is also involved.

An IDE authorizes the investigation; it does not constitute FDA approval of the device for marketing. Significant-risk studies may not begin until FDA and IRB approval have been obtained.  ￼

3. Degree of autonomy

The source needs a reproducible autonomy description, including:
	•	Tasks performed entirely by the clinician.
	•	Tasks performed by the robot.
	•	Tasks proposed by AI but confirmed by a clinician.
	•	Tasks initiated or executed autonomously.
	•	Whether the AI generates executable motion code during the procedure.
	•	Human authorization required before each motion.
	•	Manual override design.
	•	Emergency-stop design and maximum stopping time.
	•	Conditions requiring conversion to manual operation.
	•	Whether remote operation is permitted.
	•	Operator qualifications and training.

Terms such as “AI-assisted,” “semiautonomous,” and “autonomous” are insufficient without a task-level description.

4. Trial population

Specify:
	•	Cancer type, histology, stage, and anatomical location.
	•	Treatment setting and line of therapy.
	•	Adult or pediatric population.
	•	Operable or inoperable disease status.
	•	Performance status.
	•	Relevant prior therapy.
	•	Tumor-size and anatomical restrictions.
	•	Exclusion criteria based on vascular, neurological, cardiopulmonary, bleeding, or anesthesia risks.
	•	Eligibility for the conventional procedure serving as the clinical alternative.
	•	Vulnerable-population safeguards.

5. Intervention and comparator

For the initial early feasibility study:

Experimental intervention: [Device] plus the standard clinical team and standard perioperative care.

The protocol must also identify what happens when the device cannot complete its assigned task:

Rescue intervention: Immediate conversion to conventional clinician-controlled or open/laparoscopic treatment according to prespecified criteria.

For later pivotal trials, the comparator should generally be standard clinician-controlled treatment, an appropriate cleared robotic system, or a prespecified objective performance criterion. FDA recognizes that pivotal device studies may use randomized controls, concurrent nonrandomized controls, subject-as-own-control designs, or carefully justified performance criteria depending on the device and clinical question.  ￼

6. Primary early-feasibility endpoints

The source should separate clinical safety from device feasibility.

Recommended coprimary or primary endpoints include:

Safety endpoint

Incidence of device-related or procedure-related serious adverse events through [30] days.

Technical-feasibility endpoint

Proportion of procedures in which the investigational system completes all prespecified assigned tasks without an unplanned conversion caused by device malfunction or unsafe device behavior.

Necessary supporting measures include:
	•	Death.
	•	Life-threatening injury.
	•	Major bleeding.
	•	Organ or vascular injury.
	•	Unplanned conversion.
	•	Unplanned additional procedure.
	•	Device malfunction.
	•	Emergency-stop activation.
	•	Manual-override activation.
	•	Unsafe trajectory or force excursion.
	•	No-fly-zone violation.
	•	Incorrect tissue or instrument identification.
	•	Delay in treatment attributable to the system.
	•	Unanticipated adverse device effects.
	•	Procedure duration.
	•	Successful task completion.
	•	Technical success.
	•	Hospitalization and readmission.
	•	Reoperation.
	•	Cancer-specific procedural outcomes.

7. Verification and AI-specific information

The uploaded proposal’s verification framework should be translated into protocol-level fields:
	•	Prespecified verification gates.
	•	Threshold and unit for every gate.
	•	Clinical justification for every threshold.
	•	Test dataset provenance.
	•	Simulation platforms.
	•	Phantom, cadaveric, animal, and bench testing.
	•	Verification coverage.
	•	Model or software hash.
	•	Random seed where relevant.
	•	Training-data cutoff date.
	•	Version-locking procedure.
	•	Configuration-control process.
	•	Model-drift monitoring.
	•	Cybersecurity threat model.
	•	Failure-mode and effects analysis.
	•	Hazard analysis.
	•	False-negative and false-positive behavior.
	•	Subgroup performance.
	•	Human-factors and usability testing.
	•	Operator learning-curve assessment.
	•	Rules governing software changes during the study.

Material changes to software, firmware, control mechanisms, performance specifications, or basic operating principles may require an IDE supplement or other FDA notification before being deployed in the trial.  ￼

8. Safety oversight

The protocol needs:
	•	Independent medical monitor.
	•	Data and Safety Monitoring Board when justified.
	•	Staggered enrollment.
	•	Sentinel participants.
	•	Pause between initial cases.
	•	Prespecified stopping rules.
	•	Rules for cohort expansion.
	•	Device accountability.
	•	Adverse-device-effect reporting.
	•	Unanticipated adverse-device-effect reporting.
	•	Sponsor monitoring plan.
	•	Clinical-event adjudication.
	•	Independent technical review of failures.
	•	Emergency conversion procedure.
	•	Site emergency preparedness.

9. Human-subject information

Required materials include:
	•	IRB-approved protocol.
	•	Informed-consent form.
	•	Investigator agreement.
	•	Investigator qualifications.
	•	Monitoring plan.
	•	Financial disclosures.
	•	Site list.
	•	Recruitment materials.
	•	Privacy and data-use plan.
	•	Injury and compensation information.
	•	Plain-language explanation of AI and autonomy.
	•	Explanation of experimental procedures.
	•	Explanation that FDA may inspect study records.

Federal informed-consent requirements include a research statement, purpose, duration, experimental procedures, foreseeable risks, alternatives, confidentiality, injury information for more-than-minimal-risk research, contacts, voluntariness, withdrawal rights, and the approximate number of participants.  ￼

Later clinical-trial options

Traditional feasibility IDE study

Exact type

Prospective, interventional traditional feasibility study of a significant-risk investigational medical device under an FDA-approved IDE; Phase: Not Applicable.

Use

Conducted after early proof of principle, when the design is more mature, to refine procedure, patient selection, operator training, endpoints, and the design of the pivotal investigation.

Headline

[Device] Enters Traditional Feasibility IDE Study in Patients With [Cancer]

Literature

“We conducted a prospective, multicenter, open-label traditional feasibility study of a significant-risk investigational robotic device under an FDA-approved IDE.”

Do not automatically select “Device Feasibility” as the ClinicalTrials.gov primary purpose merely because “feasibility” appears in the study name. Traditional feasibility studies with at least 10 participants or health-outcome endpoints may qualify as applicable device clinical trials.  ￼

Pivotal IDE clinical investigation

Exact type

Prospective, multicenter, controlled pivotal clinical investigation of a significant-risk investigational medical device under an FDA-approved IDE; Phase: Not Applicable.

Preferred design when feasible:

Randomized, parallel-group, active-controlled, outcome-assessor-masked pivotal device trial.

Alternative when randomization is not feasible:

Prospective, multicenter, single-arm pivotal device study evaluated against a prespecified objective performance criterion.

Headline

Pivotal IDE Trial Begins Evaluation of [Device] for [Cancer Procedure]

Literature—randomized version

“We conducted a prospective, multicenter, randomized, active-controlled pivotal IDE trial comparing the [device] with standard [procedure] in patients with [cancer].”

Literature—single-arm version

“We conducted a prospective, multicenter, single-arm pivotal IDE study evaluating the [device] against a prespecified objective performance criterion.”

FDA describes pivotal studies as the definitive clinical investigations intended to support a medical-device premarket submission, while allowing the design to be tailored to the device and evidentiary question.  ￼

Postmarket clinical trial

Exact type

Prospective interventional postmarket clinical study of an FDA-cleared or FDA-approved medical device; Phase: Not Applicable.

Headline

Postmarket Trial Evaluates Long-Term Safety of [Device] in [Cancer Procedure]

Literature

“We conducted a prospective, multicenter postmarket clinical study to evaluate the long-term safety and clinical performance of the FDA-[cleared/approved] [device].”

The wording must distinguish the approved or cleared commercial configuration from any investigational software, autonomy, indication, or procedural modification.

Statutory-ready definition for §515D(i)

The missing term can be added as follows:

“The term ‘Physical AI oncology clinical investigation’ means a prospective interventional investigation, subject to part 812 of title 21, Code of Federal Regulations, in which one or more human subjects are assigned under a protocol to the use of a Physical AI system for the diagnosis, treatment, mitigation, supportive care, rehabilitation, or monitoring of cancer, for the purpose of evaluating the safety, effectiveness, feasibility, clinical performance, technical performance, or human factors of the system. Such term includes an early feasibility study, traditional feasibility study, pivotal clinical investigation, and postmarket clinical study of a device.”

For the first human study, the source should consistently use:

“First-in-human early feasibility IDE study of a significant-risk Physical AI oncology device.”

It should not use:

“Phase I robotic trial”
“IND trial”
“FDA-approved AI robot trial”
“Applicable device clinical trial”

unless the particular study independently satisfies the legal definition of that term. Small early feasibility device trials are generally excluded from the federal definition of an applicable device clinical trial, whereas later controlled or health-outcome device trials commonly qualify.  ￼
