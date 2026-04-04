"""Site systems connector: FHIR R4 client, DICOM, HL7 v2, MCP endpoints."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ProtocolType(Enum):
    FHIR_R4 = "fhir_r4"
    DICOM = "dicom"
    HL7_V2 = "hl7_v2"
    MCP = "mcp"


class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ERROR = "error"
    AUTHENTICATING = "authenticating"


@dataclass
class SystemEndpoint:
    endpoint_id: str = field(default_factory=lambda: f"EP-{uuid.uuid4().hex[:6]}")
    protocol: ProtocolType = ProtocolType.FHIR_R4
    url: str = ""
    site_id: str = ""
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    last_connected: datetime | None = None
    message_count: int = 0


@dataclass
class FHIRResource:
    resource_type: str
    resource_id: str
    data: dict[str, object] = field(default_factory=dict)


@dataclass
class HL7Message:
    message_type: str
    trigger_event: str
    segments: list[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class DICOMStudy:
    study_uid: str
    patient_id: str
    modality: str = "CT"
    series_count: int = 0
    instance_count: int = 0


class SiteSystemsConnector:
    """Connects to site systems via FHIR R4, DICOM, HL7 v2, and MCP protocols."""

    def __init__(self, site_id: str) -> None:
        self.site_id = site_id
        self.endpoints: dict[str, SystemEndpoint] = {}
        self.messages_sent: list[dict[str, str]] = []

    def register_endpoint(self, protocol: ProtocolType, url: str) -> SystemEndpoint:
        ep = SystemEndpoint(protocol=protocol, url=url, site_id=self.site_id)
        self.endpoints[ep.endpoint_id] = ep
        return ep

    def connect(self, endpoint_id: str) -> bool:
        ep = self.endpoints[endpoint_id]
        ep.status = ConnectionStatus.CONNECTED
        ep.last_connected = datetime.utcnow()
        return True

    def query_fhir(self, endpoint_id: str, resource_type: str, params: dict[str, str]) -> list[FHIRResource]:
        ep = self.endpoints[endpoint_id]
        ep.message_count += 1
        self.messages_sent.append({"protocol": "FHIR", "resource": resource_type, "params": json.dumps(params)})
        return [
            FHIRResource(resource_type=resource_type, resource_id=f"R-{i}", data={"status": "active"})
            for i in range(min(3, len(params) + 1))
        ]

    def send_hl7(self, endpoint_id: str, message_type: str, trigger: str, segments: list[str]) -> HL7Message:
        ep = self.endpoints[endpoint_id]
        ep.message_count += 1
        raw = f"MSH|^~\\&|TRIAL|{self.site_id}|EHR|{ep.url}||{message_type}^{trigger}\n"
        raw += "\n".join(segments)
        msg = HL7Message(message_type=message_type, trigger_event=trigger, segments=segments, raw=raw)
        self.messages_sent.append({"protocol": "HL7v2", "type": f"{message_type}^{trigger}"})
        return msg

    def query_dicom(self, endpoint_id: str, patient_id: str, modality: str = "CT") -> list[DICOMStudy]:
        ep = self.endpoints[endpoint_id]
        ep.message_count += 1
        return [
            DICOMStudy(
                study_uid=f"1.2.840.{uuid.uuid4().int % 10000}",
                patient_id=patient_id,
                modality=modality,
                series_count=3,
                instance_count=120,
            )
        ]

    def call_mcp_tool(self, endpoint_id: str, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        ep = self.endpoints[endpoint_id]
        ep.message_count += 1
        return {
            "tool": tool_name,
            "status": "success",
            "result": {"message": f"Executed {tool_name}", "args": arguments},
        }

    def connectivity_report(self) -> dict[str, object]:
        connected = sum(1 for e in self.endpoints.values() if e.status == ConnectionStatus.CONNECTED)
        return {
            "site_id": self.site_id,
            "endpoints": len(self.endpoints),
            "connected": connected,
            "total_messages": sum(e.message_count for e in self.endpoints.values()),
        }


def main() -> None:
    conn = SiteSystemsConnector("SITE-01")
    fhir_ep = conn.register_endpoint(ProtocolType.FHIR_R4, "https://ehr.hospital.org/fhir")
    hl7_ep = conn.register_endpoint(ProtocolType.HL7_V2, "mllp://ehr.hospital.org:2575")
    dicom_ep = conn.register_endpoint(ProtocolType.DICOM, "dicom://pacs.hospital.org:104")
    for ep in [fhir_ep, hl7_ep, dicom_ep]:
        conn.connect(ep.endpoint_id)
    patients = conn.query_fhir(fhir_ep.endpoint_id, "Patient", {"name": "Smith"})
    print(f"FHIR patients: {len(patients)}")
    msg = conn.send_hl7(hl7_ep.endpoint_id, "ADT", "A01", ["PID|1||PAT001||Smith^John"])
    print(f"HL7: {msg.message_type}^{msg.trigger_event}")
    studies = conn.query_dicom(dicom_ep.endpoint_id, "PAT001")
    print(f"DICOM studies: {len(studies)}")
    print(f"Report: {conn.connectivity_report()}")


if __name__ == "__main__":
    main()
