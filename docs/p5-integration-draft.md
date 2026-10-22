# P5 연계 초안 — 스케줄링 결과 → 자원 풀 · KETRIS (안 v0.1, 2026-10-14)

> **진행 (10-14):** 자원 풀 경로는 **구현됨** — keti-scheduler 0.3 이 GPU(LSU) 승인 시 `UpdateAllocationDecision(RESERVED)`,
> Kueue Workload 종료·삭제 시 `RELEASED` 를 호출하고 결과를 결정 이력 `resource_pool` 에 남긴다 (pod_name = decision_id).
> NPU·CPU 승인은 `NOT_APPLICABLE`. RPM 무응답이면 승인은 그대로, 레코드에 `*_ERROR`.
> RPM 은 232 호스트 프로세스(`resource-pool-manager`, gRPC :50052, `SLICE_CR_WATCH=0`)로 기동 — 재부팅 시 다시 띄워야 한다.
> 검증: 호스트에서 RPM 실호출 RESERVED/RELEASED 성공, 스케줄러 Pod → RPM 연결, NPU 승인 NOT_APPLICABLE 기록.
> 클러스터 안 GPU 승인 → RESERVED 는 LSU 를 풀링 실험이 점유 중이라 시간 조율 후 확인. KETRIS(SliceAllocation) 경로는 아래 합의 대기.

**결론:** 새 계약을 만들지 않고 기존 두 경로를 그대로 쓴다. keti-scheduler 가 승인(APPROVED)할 때
① 자원 풀에 `UpdateAllocationDecision(RESERVED)`, ② KETRIS 에 `SliceAllocation` spec 을 쓰고,
KETRIS 가 갱신한 status 를 결정 이력의 `applied` 에 붙인다 (p.42 "계획 대비 실제").
**KETRIS 명령 세트 담당자와 필드 합의 전 — 아래 매핑은 KETI 안이다.**

## 1. 흐름

| 시점 | keti-scheduler | 자원 풀 (gRPC C-RP-03) | KETRIS (SliceAllocation CR) |
|---|---|---|---|
| 승인 | DecisionRecord APPROVED | `UpdateAllocationDecision` state=RESERVED | spec 생성 |
| 적용 | `applied` 채움 | – | status.phase=PLACED, `applied`, `occupancy` |
| 종료·삭제 | 레코드 유지 | `UpdateAllocationDecision` state=RELEASED | CR 삭제 → status RELEASED |
| 보류 | DecisionRecord HELD | 호출 없음 | 쓰지 않음 |

## 2. 필드 매핑

### 2.1 DecisionRecord → `UpdateAllocationDecisionRequest` (resource_pool.proto, 확정 계약 — 변경 없음)

| 요청 필드 | 값 |
|---|---|
| `workload_id` | `DecisionRecord.workload_id` (keti.re.kr/workload-id). 기존 주석 "AIWorkload metadata.uid (데모: pod_name)" → **workload-id 로 통일 제안** |
| `device_id` | 계획 장치 ID — 지금 골격은 노드 단위까지만 정함 (아래 §3-1) |
| `partition_id` | partition_set.proto `Partition.partition_id` |
| `lsu_amount` | `plan.amount` (자원이 keti.re.kr/lsu 일 때) |
| `allocation_state` | RESERVED / RELEASED |
| `model_id` | ExecEnvInfo `model.model_id` (모델 등록 연계 전에는 빈 값) |
| `pod_name` | `applied.pods[0]` (참고용) |

### 2.2 DecisionRecord → `SliceAllocation.spec` (C-IS-08 확정 스키마 — 변경 없음)

| spec 필드 | 값 |
|---|---|
| `gpuId` | 계획 장치 ID (§3-1) |
| `migPartitionId` | 성균관대 계획안이 MIG 를 지정한 경우만 (`PlanCandidate.partition_id`) |
| `lsuAmount` | `plan.amount` |
| `slo.type / target / priority` | annotation `keti.re.kr/slo-latency-ms` → LATENCY, `slo-throughput` → THROUGHPUT, priority 는 PriorityClass 값 |
| `profile.workloadType` | `input.workload_kind` (TRAIN / INFER / PREPROC — 같은 값 집합) |
| `profile.computeBoundness / memoryBoundness / estPeakMemMb` | 연세대 프로파일 (ModelExecutionHistory 확장 10~12번) — 없으면 생략 |
| `residual.freeSmRatio / freeTimeRatio / freeMemMb` | 판단 시점 `input.nodes` 의 여유 (partition_set.proto `DeviceUsage`) |

### 2.3 `SliceAllocation.status` → DecisionRecord `applied`

| status | applied |
|---|---|
| `phase` | PLACED 면 적용 완료, FAILED 면 실패로 기록 |
| `applied.spaceRatio / timeRatio / memMb` | p.42 "실제 적용" 열 (공간 구성 · 시간비율 · 공간 메모리) |
| `timestamp` | `applied.observed_at` |
| (없음) | `ketris_config_version` — p.42 "분할 설정 v2" 에 해당하는 필드가 status 에 없다 → §3-2 |

## 3. KETRIS 담당자와 맞출 것

1. **장치 단위 계획:** 지금 스케줄러는 노드까지만 고른다. `gpuId` 를 채우려면 장치(GPU UUID)까지 골라야 한다.
   KETRIS 가 노드 안 장치 선택을 맡을지, 스케줄러가 정할지.
2. **분할 설정 버전:** p.42 는 "분할 설정 v2" 를 표시한다. status 에 `configVersion` 필드 추가 가능 여부.
3. **NPU:** SliceAllocation 은 GPU 전용(`gpuId`, LSU)이다. NPU-PE 승인 결과를 KETRIS 로 넘길지, 자원 풀에만 기록할지.
4. **CR 이름·네임스페이스:** 어댑터 기준 `edge-ai` 네임스페이스. 이름을 `<workload-id>-<decision-id>` 로 제안 (Pod 단위 Workload 가 여러 개인 서비스 대응).
5. **CRD 설치:** 현재 클러스터에 `sliceallocations.keti.re.kr` 가 없다. 설치 시점(풀링 과제 일정)과 `tools/ketris_cr_adapter.py` 재가동 여부.

## 4. 구현 순서 (합의 후)
1. 자원 풀 호출 (RESERVED / RELEASED) — 계약이 확정돼 있어 먼저 가능
2. SliceAllocation spec 쓰기 + status watch → `applied` 갱신
3. 장치 단위 계획 (§3-1 합의 결과에 따라)
