## ChatGPT-5.5-Thinking-Extended-19Jun26

### Prompts
- Only provide clinical trial options. Provide the number of times each were used in the last 5 years for cancer treatments. Further detail exactly what each is, and how to state each in headlines and literature. Expanded Access (Compassionate Use): This is the most common pathway. It is an FDA-regulated process that allows patients with immediately life-threatening conditions or serious diseases to gain access to an investigational medical product (drug, biologic, or medical device) outside of a clinical trial. The patient's doctor must apply to the FDA and the drug manufacturer, arguing that the potential patient benefit justifies the potential risks of the treatment.
Emergency IND (Investigational New Drug): A specific type of Expanded Access used in urgent, life-threatening situations where there isn't enough time to go through the standard Compassionate Use review process. The FDA can grant authorization over the phone.
- Provide the exact trial type and information needed for the following.
- Provide same output, but with all references as bibtex entries at the end (with reference numbers in the body corresponding to the numbers for each reference at the bottom). 

### Output
Exact classification and required trial information

Reference numbers in brackets correspond to the numbered BibTeX entries at the end.

Exact classification of the uploaded proposal

The uploaded source does not yet define an exact clinical-trial type. It repeatedly uses “Physical AI oncology clinical investigation,” while:
	•	Section 4 characterizes the robot-control software as a medical-device function.
	•	Proposed §515D(j) preserves the investigational device exemption framework.
	•	Proposed §515D(i) does not define “Physical AI oncology clinical investigation.”
	•	References to the IND framework apply only when an investigational drug or biologic is also being studied; they do not convert a robotic-device investigation into an IND trial. [1]

The appropriate umbrella classification is:

Prospective interventional medical-device clinical investigation conducted under 21 CFR part 812.

It should ordinarily not be described as a Phase I, Phase II, or Phase III trial. ClinicalTrials.gov uses Phase: Not Applicable for device trials because the FDA-defined phase categories apply to drugs and biological products. [2,4]  ￼

⸻

Exact initial trial type

For the first patient-facing investigation contemplated by the proposal, the recommended designation is:

A prospective, interventional, open-label, single-group, first-in-human early feasibility study of a significant-risk investigational medical device, conducted under an FDA-approved investigational device exemption; Phase: Not Applicable; Primary Purpose: Device Feasibility.

FDA describes an early feasibility study as an early clinical evaluation of a device that provides initial clinical-safety, functionality, and proof-of-principle information. It is especially appropriate when clinical experience is necessary because nonclinical methods alone cannot provide the information needed to continue development. Early feasibility studies involving significant-risk devices require an IDE. [3]  ￼

The following elements are recommended design characteristics, rather than universal regulatory requirements:
	•	Prospective.
	•	Interventional.
	•	Single group.
	•	Open label.
	•	Staggered or sentinel enrollment.
	•	Independent outcome assessment when feasible.
	•	Prespecified progression, pause, and stopping rules.

Exact wording for a protocol

This is a prospective, first-in-human, open-label, single-group early feasibility clinical investigation of the [full device name] Physical AI robotic system in adults undergoing [specific oncology procedure]. The investigation will be conducted as a significant-risk device study under an FDA-approved investigational device exemption.

Exact wording for a headline

First-in-Human Early Feasibility IDE Study Begins for [Device] in [Cancer Procedure]

Alternative:

FDA Approves IDE for Early Feasibility Study of [Device] in Patients With [Cancer]

Do not write:

Phase I trial of the AI robot
IND trial of the robotic system
FDA-approved robotic cancer treatment
FDA approves the device for cancer treatment

An approved IDE authorizes the clinical investigation; it does not constitute marketing authorization or a determination that the device is safe and effective for commercial use. [2,5]  ￼

⸻

Significant-risk designation

Based on the proposal’s descriptions of:
	•	Patient-contact robotic motion.
	•	Surgical or interventional functions.
	•	Needle or instrument placement.
	•	Force control.
	•	Vascular and anatomical exclusion zones.
	•	Emergency stops.
	•	Potentially catastrophic motion or control failures.

the appropriate presumptive classification is:

Significant-risk investigational device study.

Under 21 CFR 812.3(m), a significant-risk device includes an investigational device used for a substantially important diagnostic or therapeutic purpose when the device presents a potential for serious risk to a subject’s health, safety, or welfare. [2]  ￼

This classification is a reasoned regulatory recommendation, not a binding determination. The reviewing IRB initially evaluates the risk classification, subject to FDA’s authority to make the final determination.

Exact literature wording

The investigational robotic system was treated as a significant-risk medical device under 21 CFR part 812.

More cautious wording before a formal determination:

Based on its intended therapeutic functions and potential for serious harm in the event of unsafe operation, the system was presumptively classified as a significant-risk investigational device, subject to confirmation by the reviewing IRB and FDA.

⸻

Exact ClinicalTrials.gov entries

Registration field	Recommended entry
Study Type	Interventional
Study Phase	Not Applicable
Primary Purpose	Device Feasibility
Intervention Model	Single Group Assignment
Allocation	N/A
Masking	None—Open Label
Independent assessment	Masked independent outcome assessor when feasible; describe under outcome-assessment procedures
Number of Arms	1
Arm Type	Experimental
Intervention Type	Device
Studies a U.S. FDA-Regulated Device Product	Yes
Device Product Not Approved or Cleared by FDA	Yes, assuming no studied configuration or intended use has been cleared or approved
U.S. FDA IND or IDE	Yes
FDA Center	CDRH, unless another FDA center has jurisdiction
IND/IDE Number	Enter the assigned IDE number; this field is administrative and is not publicly displayed
First-in-Human status	State in the official title and brief summary; ClinicalTrials.gov does not provide a separate public “First-in-Human” yes/no field
Official Title	A Prospective, First-in-Human Early Feasibility Study of the [Device Name] Physical AI Robotic System for [Procedure] in Adults With [Cancer Type]
Brief Title	Early Feasibility Study of [Device] in [Cancer Type]
Condition or Disease	Exact cancer diagnosis, histology, stage, and anatomical site
Intervention Name	Full device name, model, software version, and autonomy configuration
Enrollment	A small cohort justified by the safety and feasibility objectives
Study Sites	Every participating institution
Follow-up	Acute, 30-day, and procedure-appropriate longer-term follow-up

ClinicalTrials.gov defines Device Feasibility as evaluation of a device in a small clinical trial—generally fewer than 10 participants—or testing of a prototype where the primary outcome concerns feasibility rather than health outcomes. “Phase: Not Applicable” applies to trials without FDA-defined drug phases, including device investigations. [4]  ￼

A sequential or staged model may be more appropriate than a single undifferentiated cohort when the protocol includes progressively greater autonomy, such as:
	1.	Clinician-controlled robotic assistance.
	2.	Clinician-supervised semiautonomous operation.
	3.	Higher-autonomy operation only after independent safety review.

In that case, the protocol should still ordinarily be registered as one study, with the stages described as cohorts, arms, or prespecified enrollment stages according to how participants are assigned.

⸻

Information that must be specified

1. Device identity

The protocol, IDE application, registry record, investigator materials, and proposed statutory record should specify:
	•	Device trade name.
	•	Device generic name.
	•	Manufacturer.
	•	Legal sponsor.
	•	Hardware model.
	•	Hardware configuration.
	•	Software version.
	•	Firmware version.
	•	AI-model version.
	•	Algorithm version.
	•	Configuration identifier or software hash.
	•	Intended use.
	•	Exact oncology indication.
	•	Exact clinical procedure.
	•	Patient-contact components.
	•	Instruments controlled by the system.
	•	Whether the device is reusable, disposable, implanted, or procedure limited.
	•	Whether the device is a new product or a modification of a legally marketed device.
	•	Anticipated 510(k), De Novo, PMA, or HDE pathway.

An IDE application must contain the device description, intended use, principles of operation, prior investigations, investigational plan, manufacturing information, labeling, informed-consent materials, investigator information, and IRB information. [2,6]  ￼

Exact protocol wording

The investigational intervention is the [manufacturer] [device name], model [number], incorporating hardware configuration [identifier], firmware version [version], software version [version], and AI-model version [version/hash]. The system is intended in this investigation to perform or assist with [precisely enumerated tasks] during [procedure] in patients with [cancer diagnosis and stage].

⸻

2. Investigational status

State explicitly:

The [device name] is an investigational medical device. It has not been cleared or approved by the U.S. Food and Drug Administration for the investigational use evaluated in this study.

The record should also identify:
	•	Significant-risk determination.
	•	IDE application number.
	•	FDA IDE status.
	•	Date of FDA authorization.
	•	IRB approval status at every site.
	•	Q-Submission or Pre-Submission number.
	•	Whether any component is already marketed.
	•	Whether the investigational use differs from an existing cleared or approved use.
	•	Whether a drug, biologic, contrast agent, imaging agent, or other combination-product component is included.

A significant-risk investigation may not begin until both FDA and the reviewing IRB have authorized it. [2,5,6]  ￼

⸻

3. Degree of autonomy

The protocol must define autonomy at the task level, rather than relying only on labels such as “AI assisted,” “semiautonomous,” or “autonomous.”

Specify:
	•	Tasks performed solely by the clinician.
	•	Tasks proposed by the AI.
	•	Tasks requiring clinician confirmation.
	•	Tasks initiated by the robot.
	•	Tasks executed autonomously.
	•	Whether executable motion code is generated intraoperatively.
	•	Whether each motion requires prospective clinician authorization.
	•	What information is displayed to the operator before authorization.
	•	Manual-override mechanisms.
	•	Emergency-stop mechanisms.
	•	Maximum stopping time.
	•	Maximum stopping distance.
	•	Safe-state definition.
	•	Conditions requiring conversion to manual operation.
	•	Whether the system may modify a treatment plan during the procedure.
	•	Whether the device learns or adapts during use.
	•	Whether remote operation is permitted.
	•	Operator qualifications.
	•	Required training and credentialing.

Exact protocol wording

The investigational system operates at a task-specific autonomy level. It may independently perform [tasks], may recommend but may not execute [tasks] without clinician confirmation, and may not independently perform [prohibited tasks]. The responsible clinician retains authority to pause, override, or terminate system operation at all times.

⸻

4. Trial population

Specify:
	•	Cancer type.
	•	Histology.
	•	Molecular subtype when relevant.
	•	Stage.
	•	Anatomical location.
	•	Treatment setting.
	•	Line of therapy.
	•	Operability.
	•	Performance status.
	•	Prior treatments.
	•	Tumor dimensions.
	•	Anatomical restrictions.
	•	Distance from critical structures.
	•	Bleeding risk.
	•	Anesthesia eligibility.
	•	Cardiopulmonary restrictions.
	•	Neurological restrictions.
	•	Vascular restrictions.
	•	Eligibility for the conventional procedure.
	•	Adult or pediatric status.
	•	Vulnerable-population safeguards.

Do not use “oncology patients” as the complete population description.

Exact eligibility wording

Adults aged [range] with histologically confirmed [cancer type and subtype], clinical stage [stage], requiring [procedure], whose tumor measures [range] and is located at least [distance] from [critical structure], and who are medically eligible for both the investigational procedure and the prespecified rescue procedure.

⸻

5. Intervention and comparator

Initial early feasibility study

The experimental intervention should be defined as:

[Device] plus the standard clinical team, standard anesthesia, standard imaging, and standard perioperative care.

The rescue intervention should be defined separately:

Immediate conversion to conventional clinician-controlled, laparoscopic, endoscopic, percutaneous, or open treatment under prespecified conversion criteria.

The conventional clinical team remains part of the investigational intervention and should not be described as absent merely because the robot performs certain tasks.

Comparator

A separate control arm is not ordinarily necessary for the first small early feasibility investigation when the primary objectives are initial safety and technical feasibility. Later studies may use:
	•	Randomized active control.
	•	Concurrent nonrandomized control.
	•	Matched control.
	•	Within-subject control.
	•	Historical control.
	•	Objective performance criterion.

The choice must be justified according to the device, procedure, disease, endpoint, and intended marketing claim. [3,7]  ￼

⸻

6. Primary early-feasibility endpoints

Clinical safety and device feasibility should be separated.

Recommended primary safety endpoint

Incidence of device-related or procedure-related serious adverse events through 30 days after the investigational procedure.

The time period should be modified where the procedure’s clinically important risks extend beyond 30 days.

Recommended primary technical-feasibility endpoint

Proportion of procedures in which the investigational system completes all prespecified assigned tasks without an unplanned conversion caused by device malfunction, unsafe device behavior, or inability of the system to perform the assigned task.

Necessary supporting endpoints
	•	All-cause mortality.
	•	Device-related mortality.
	•	Life-threatening injury.
	•	Major bleeding.
	•	Organ injury.
	•	Vascular injury.
	•	Neurological injury.
	•	Thermal injury.
	•	Unplanned additional procedure.
	•	Unplanned conversion.
	•	Device malfunction.
	•	Software failure.
	•	System restart.
	•	Emergency-stop activation.
	•	Manual-override activation.
	•	Unsafe trajectory.
	•	Force-limit excursion.
	•	No-fly-zone violation.
	•	Incorrect tissue identification.
	•	Incorrect instrument identification.
	•	Loss of tracking.
	•	Communication failure.
	•	Treatment delay attributable to the device.
	•	Unanticipated adverse device effect.
	•	Procedure duration.
	•	Device task-completion time.
	•	Successful device setup.
	•	Successful calibration.
	•	Technical success.
	•	Hospital length of stay.
	•	Readmission.
	•	Reoperation.
	•	Cancer-specific procedural outcomes.

Exact endpoint wording

The primary technical-feasibility endpoint is successful completion of all protocol-assigned robotic tasks without device-related unplanned conversion, emergency intervention, or clinically significant violation of a prespecified safety boundary.

“Technical success” should not be used without a complete operational definition.

⸻

7. Verification and AI-specific information

The proposal’s verification framework should be converted into testable protocol fields:
	•	Prespecified verification gates.
	•	Pass/fail threshold for every gate.
	•	Measurement unit for every threshold.
	•	Clinical justification for every limit.
	•	Test-dataset provenance.
	•	Dataset inclusion and exclusion criteria.
	•	Simulation environment.
	•	Phantom testing.
	•	Cadaveric testing.
	•	Animal testing.
	•	Bench testing.
	•	Verification coverage.
	•	Software hash.
	•	Model hash.
	•	Random seed where relevant.
	•	Training-data cutoff date.
	•	Model-locking procedure.
	•	Version-control procedure.
	•	Configuration-control procedure.
	•	Model-drift monitoring.
	•	Cybersecurity threat model.
	•	Failure-mode and effects analysis.
	•	Hazard analysis.
	•	False-positive performance.
	•	False-negative performance.
	•	Subgroup performance.
	•	Human-factors testing.
	•	Operator learning-curve assessment.
	•	Rules for software changes during the trial.

Material changes to the device, software, control mechanism, performance specification, manufacturing process, or investigational plan may require an IDE supplement and appropriate FDA and IRB authorization before implementation. [2,6]  ￼

Exact protocol wording

The investigational configuration shall remain locked during each enrollment stage. No participant shall be treated using a software, firmware, hardware, or model configuration that differs from the configuration approved for that stage unless the change has been reviewed and authorized under the applicable IDE and IRB change-control procedures.

⸻

8. Safety oversight

The protocol should include:
	•	Independent medical monitor.
	•	Data and Safety Monitoring Board when justified.
	•	Sentinel enrollment.
	•	Staggered enrollment.
	•	Prespecified waiting period between initial cases.
	•	Case-by-case review before progression.
	•	Enrollment-pause criteria.
	•	Study-stopping criteria.
	•	Cohort-expansion criteria.
	•	Device accountability.
	•	Unanticipated adverse-device-effect reporting.
	•	Sponsor monitoring.
	•	Clinical-event adjudication.
	•	Independent technical failure review.
	•	Emergency conversion procedures.
	•	Site emergency-preparedness requirements.
	•	Rules for restarting after a pause.

Example stopping rule

Enrollment shall be paused upon any device-related death, any unanticipated adverse device effect involving a life-threatening event, any repeated violation of a critical safety boundary, or two device-related serious adverse events of the same clinically meaningful type. Enrollment may resume only after sponsor, medical-monitor, DSMB, IRB, and FDA review as applicable.

⸻

9. Human-subject information

Required materials include:
	•	IRB-approved protocol.
	•	Informed-consent form.
	•	Investigator agreement.
	•	Investigator qualifications.
	•	Operator-training records.
	•	Monitoring plan.
	•	Financial disclosures.
	•	Site list.
	•	Recruitment materials.
	•	Privacy plan.
	•	Data-use plan.
	•	Injury and compensation information.
	•	Plain-language explanation of AI.
	•	Plain-language explanation of autonomy.
	•	Explanation of experimental tasks.
	•	Explanation of foreseeable device failures.
	•	Explanation of conventional alternatives.
	•	Explanation of rescue procedures.
	•	Explanation that FDA may inspect study records.

Federal informed-consent requirements include a statement that the activity is research, its purpose, expected duration, procedures, experimental procedures, foreseeable risks, expected benefits, alternatives, confidentiality, research-injury information for more-than-minimal-risk investigations, contacts, voluntary participation, and withdrawal rights. [8]  ￼

Exact consent wording

The robotic system is investigational. Some actions performed during the procedure may be selected, planned, recommended, or executed by software using artificial-intelligence methods. The study team does not know whether use of the system is safer, as safe as, or more effective than the conventional procedure.

⸻

Later clinical-trial options

1. Traditional feasibility IDE study

Exact type

Prospective, interventional traditional feasibility study of a significant-risk investigational medical device under an FDA-approved IDE; Phase: Not Applicable.

Purpose

This study follows initial proof of principle and may be used to refine:
	•	Device design.
	•	Procedure.
	•	Patient selection.
	•	Operator training.
	•	Endpoint definitions.
	•	Site requirements.
	•	Pivotal-study design.

Headline

[Device] Enters Traditional Feasibility IDE Study in Patients With [Cancer]

Literature wording

We conducted a prospective, multicenter, open-label traditional feasibility study of a significant-risk investigational robotic device under an FDA-approved IDE.

“Device Feasibility” should be selected as the ClinicalTrials.gov primary purpose only when the study is genuinely a small feasibility or prototype study centered on feasibility rather than health outcomes. A study is not automatically a feasibility study merely because “feasibility” appears in its title. [4,9,10]  ￼

⸻

2. Pivotal IDE clinical investigation

Preferred exact type

Prospective, multicenter, randomized, parallel-group, active-controlled, outcome-assessor-masked pivotal clinical investigation of a significant-risk investigational medical device under an FDA-approved IDE; Phase: Not Applicable.

Alternative when randomization is not feasible

Prospective, multicenter, single-arm pivotal IDE study evaluated against a prespecified objective performance criterion.

Headline

Pivotal IDE Trial Begins Evaluation of [Device] for [Cancer Procedure]

Literature wording—randomized design

We conducted a prospective, multicenter, randomized, active-controlled pivotal IDE trial comparing the [device] with standard [procedure] in patients with [cancer].

Literature wording—single-arm design

We conducted a prospective, multicenter, single-arm pivotal IDE study evaluating the [device] against a prespecified objective performance criterion.

Pivotal investigations are intended to provide definitive clinical evidence supporting a device premarket submission. FDA recognizes randomized controls, concurrent nonrandomized controls, subject-as-own-control approaches, and appropriately justified performance criteria, depending on the clinical and regulatory question. [7]  ￼

⸻

3. Postmarket interventional clinical trial

Exact type

Prospective, interventional postmarket clinical study of an FDA-cleared or FDA-approved medical device; Phase: Not Applicable.

Headline

Postmarket Trial Evaluates Long-Term Safety of [Device] in [Cancer Procedure]

Literature wording

We conducted a prospective, multicenter postmarket clinical study to evaluate the long-term safety and clinical performance of the FDA-[cleared/approved] [device].

The literature must distinguish:
	•	The cleared or approved commercial configuration.
	•	Investigational software modifications.
	•	Investigational autonomy functions.
	•	New cancer indications.
	•	New procedural uses.
	•	New patient populations.

A commercially marketed base device may still be investigational when studied with a materially different intended use, configuration, software function, or clinical claim. [5]  ￼

⸻

Statutory-ready definition for proposed §515D(i)

The missing definition may be drafted as follows:

“The term ‘Physical AI oncology clinical investigation’ means a prospective interventional investigation, subject to part 812 of title 21, Code of Federal Regulations, in which one or more human subjects are prospectively assigned under a protocol to the use of a Physical AI system for the diagnosis, treatment, mitigation, supportive care, rehabilitation, or monitoring of cancer, for the purpose of evaluating the safety, effectiveness, feasibility, clinical performance, technical performance, or human factors of the system. Such term includes an early feasibility study, traditional feasibility study, pivotal clinical investigation, and interventional postmarket clinical study of a device.”

To avoid unintentionally covering observational studies, the phrase “prospectively assigned” should remain in the definition.

For the first human study, the proposal should consistently use:

First-in-human early feasibility IDE study of a significant-risk Physical AI oncology device.

It should not use:

Phase I robotic trial
IND trial
FDA-approved AI robot trial
FDA-approved cancer robot
Applicable device clinical trial

unless the particular investigation independently satisfies the legal definition of an applicable device clinical trial. Small feasibility and prototype-device studies centered on feasibility rather than health outcomes are generally excluded from that definition; later controlled studies of health outcomes may qualify. [4,9,10]  ￼

BibTeX references

% [1]
@misc{kawchak2026physicalai,
  author       = {Kawchak, Kevin},
  title        = {{Verification Before Generation in Physical AI Oncology Trials
                   Act of 2026 (H.R. 9510), Bill v5.0: A Financial Data
                   Amendment to the Federal Food, Drug, and Cosmetic Act}},
  year         = {2026},
  month        = jun,
  day          = {10},
  version      = {5.0},
  doi          = {10.5281/zenodo.20619762},
  url          = {https://doi.org/10.5281/zenodo.20619762},
  note         = {Independent research draft; uploaded LaTeX source files}
}

% [2]
@misc{ecfr2026part812,
  author       = {{Electronic Code of Federal Regulations}},
  title        = {{21 CFR Part 812---Investigational Device Exemptions}},
  organization = {{Office of the Federal Register}},
  year         = {2026},
  url          = {https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-812},
  urldate      = {2026-06-19},
  note         = {Title 21 current through June 17, 2026}
}

% [3]
@techreport{fda2013earlyfeasibility,
  author       = {{U.S. Food and Drug Administration}},
  title        = {{Investigational Device Exemptions (IDEs) for Early
                   Feasibility Medical Device Clinical Studies, Including
                   Certain First in Human (FIH) Studies: Guidance for Industry
                   and Food and Drug Administration Staff}},
  institution  = {{Center for Devices and Radiological Health}},
  year         = {2013},
  month        = oct,
  number       = {FDA-2011-D-0787},
  type         = {Guidance Document},
  url          = {https://www.fda.gov/regulatory-information/search-fda-guidance-documents/investigational-device-exemptions-ides-early-feasibility-medical-device-clinical-studies-including},
  urldate      = {2026-06-19},
  note         = {Final guidance}
}

% [4]
@misc{clinicaltrialsgov2025protocoldefinitions,
  author       = {{National Library of Medicine}},
  title        = {{Protocol Registration Data Element Definitions for
                   Interventional and Observational Studies}},
  organization = {{ClinicalTrials.gov}},
  year         = {2025},
  month        = apr,
  day          = {24},
  url          = {https://clinicaltrials.gov/policy/protocol-definitions},
  urldate      = {2026-06-19}
}

% [5]
@misc{fda2024ideoverview,
  author       = {{U.S. Food and Drug Administration}},
  title        = {{Investigational Device Exemption (IDE)}},
  organization = {{Center for Devices and Radiological Health}},
  year         = {2024},
  month        = aug,
  day          = {22},
  url          = {https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/investigational-device-exemption-ide},
  urldate      = {2026-06-19},
  note         = {Content current as of August 22, 2024}
}

% [6]
@misc{fda2026ideapplication,
  author       = {{U.S. Food and Drug Administration}},
  title        = {{IDE Application}},
  organization = {{Center for Devices and Radiological Health}},
  year         = {2026},
  url          = {https://www.fda.gov/medical-devices/investigational-device-exemption-ide/ide-application},
  urldate      = {2026-06-19},
  note         = {Living FDA regulatory-information webpage}
}

% [7]
@techreport{fda2013pivotaldesign,
  author       = {{U.S. Food and Drug Administration}},
  title        = {{Design Considerations for Pivotal Clinical Investigations
                   for Medical Devices: Guidance for Industry, Clinical
                   Investigators, Institutional Review Boards and FDA Staff}},
  institution  = {{Center for Devices and Radiological Health}},
  year         = {2013},
  month        = nov,
  number       = {FDA-2011-D-0567},
  type         = {Guidance Document},
  url          = {https://www.fda.gov/regulatory-information/search-fda-guidance-documents/design-considerations-pivotal-clinical-investigations-medical-devices},
  urldate      = {2026-06-19},
  note         = {Final guidance}
}

% [8]
@misc{ecfr2026part50,
  author       = {{Electronic Code of Federal Regulations}},
  title        = {{21 CFR Part 50---Protection of Human Subjects}},
  organization = {{Office of the Federal Register}},
  year         = {2026},
  url          = {https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-50},
  urldate      = {2026-06-19},
  note         = {See especially 21 CFR 50.20--50.27; Title 21 current through
                  June 17, 2026}
}

% [9]
@misc{ecfr2026part11,
  author       = {{Electronic Code of Federal Regulations}},
  title        = {{42 CFR Part 11---Clinical Trials Registration and Results
                   Information Submission}},
  organization = {{Office of the Federal Register}},
  year         = {2026},
  url          = {https://www.ecfr.gov/current/title-42/chapter-I/subchapter-A/part-11},
  urldate      = {2026-06-19}
}

% [10]
@misc{clinicaltrialsgov2026faq,
  author       = {{National Library of Medicine}},
  title        = {{Frequently Asked Questions}},
  organization = {{ClinicalTrials.gov}},
  year         = {2026},
  month        = may,
  day          = {6},
  url          = {https://clinicaltrials.gov/policy/faq},
  urldate      = {2026-06-19}
}
