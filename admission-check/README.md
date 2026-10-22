# keti-scheduler — Kueue AdmissionCheck 컨트롤러 (P4 골격, 안 v0.1)

> 개발 검토용 코드입니다. 2026-10-22 업로드 검토에서 배치 제약과 예약 정리의 미해결 문제가 확인되었습니다. 아래 과거 실험 기록은 모든 운영 조건의 정확성을 보장하지 않습니다. `main` 병합·운영 반영 전 [검토 결과](PUBLICATION_REVIEW.md)의 항목을 해결해야 합니다.

Kueue 가 쿼터를 잡은 Workload 를 Filter → Score 로 승인·보류하고, 판단마다 결정 이력을 남긴다.
메시지 정의 `../proto/scheduler_decision.proto`, 화면은 exec-env `keti-queue-api` 의 `/api/v1/decisions`.

| 단계 | 내용 |
|---|---|
| Filter | 배정된 ResourceFlavor 노드 중 요청 자원이 **물리적으로** 남은 노드 (allocatable − 올라간 Pod 요청) |
| Score (KETI) | `W_FREE`×배치 후 여유 비율 + `W_FRAG`×(1 − 배치 후 단편화). 기본 0.4 / 0.6. SLO 여유는 데이터 없으면 제외 |
| 성균관대 1·2회차 | `fetch_plan()` 자리만 있음 — 형식 회신 전까지 None → KETI 스코어로 판단 |
| 승인 | AdmissionCheck `Ready` + podSetUpdates (계획 노드 nodeSelector, `keti.re.kr/decision-id` 라벨) |
| 공간 부족 | `Retry` + `requeueAfterSeconds`(기본 20) — 쿼터 반납 후 재대기, 사유 `RESOURCE_WAIT` |
| 프로파일 대기 | `Pending`(쿼터 유지), 사유 `PROFILE_WAIT`. 데모: Pod 템플릿 annotation `keti.re.kr/profile-wait-seconds` |
| 적용 결과 | 승인 후 `decision-id` 라벨 Pod 의 실제 노드를 같은 레코드에 붙임 (p.42 계획 대비 실제) |

배포: `deploy/scheduler.yaml` (keti-exec-system, 결정 이력 PVC local-path). **이 파일만으로는 어떤 큐에도 붙지 않는다.**
운영 큐(gpu-cq/npu-cq)에 붙이는 것은 ClusterQueue `spec.admissionChecksStrategy` 변경 — 승인 필요.
검증용 큐: `deploy/test-queue.yaml` (cohort 없음).

## 검증 (2026-10-14, sched-test-cq)

| 경우 | 결과 |
|---|---|
| LSU 10 Job | 노드 LSU allocatable 이 잠깐 0 이 된 동안 RESOURCE_WAIT 보류 6회 → 복구 후 승인, 계획 노드 = 실제 노드 |
| LSU 300 Job (쿼터 400 > 물리 240) | Kueue 는 쿼터 확보, 스케줄러가 물리 부족으로 Retry — 20초 간격 재판정 |
| profile-wait 20s Job | PROFILE_WAIT 보류(이력 1줄, 반복 횟수만 증가) → 20초 후 KETI 스코어 폴백 승인 |
| NPU-PE Deployment 2 Pod | Pod 별 Workload 각각 판단, 승인분은 적용 결과 기록 |

| (0.2) NPU-PE 4 Job 3개 동시 제출 (물리 PE 8) | 2개 승인, 1개 RESOURCE_WAIT — 승인 후 Pod 가 붙기 전 몫을 예약으로 셈(`INFLIGHT_SECONDS`, 기본 180) |
| (0.2, 운영) 큐 라벨 없는 NPU Job, edge-dev | 라우터 → npu-lq → npu-cq → keti-scheduler 승인, 계획 노드 = 실제 노드 |

관찰: Kueue를 거치지 않는 외부 워크로드의 자원 점유는 Kueue 쿼터만으로 확인할 수 없습니다. 물리 여유 Filter의 필요성을 보여주는 과거 실험 관찰이며, 현재 노드 상태를 나타내지 않습니다.

## 환경별 설정
`deploy/scheduler.yaml`은 배포 템플릿입니다. `RPM_ADDR`은 빈 값으로 제공하며, 자원 풀 연계가 필요하면 대상 환경의 주소를 별도로 설정해야 합니다. 실제 ClusterQueue 연결과 운영 반영은 이 Draft PR의 범위에 포함하지 않습니다.
