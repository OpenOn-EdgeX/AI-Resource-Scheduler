# structure set (KETI 안 v0.1, 2026-10-14)

기관 간 주고받는 자료의 구조. **JSON 이 정본**이고, `../proto/*.proto` 는 같은 필드 이름의 전송 구현이다.
각 구조 = 필드 정의(`*.json`) + 예시(`*.example.json`). 정의의 필드 표기: `type` / `required` / `desc` / `source`(값 출처) / `now`(현재 채워지는지) / `map_to`(K8s 반영 위치).

공개 저장소의 예제는 **익명화한 형식 예시**입니다. 노드·장치·워크로드 식별자, 시각, 배포 정보는 예시 값이며 현재 운영 환경을 나타내지 않습니다. 과거 기록에서 남긴 수치도 새로운 실험의 측정 결과로 사용하면 안 됩니다.

| 구조 | 방향 | 정의 | 예시 |
|---|---|---|---|
| 실행환경 정보 | KETI → 연세대 → 성균관대 | exec-env-info.json | 익명화한 실행환경 형식 예시 |
| 파티션 정보 세트 | KETI → 성균관대 | partition-set.json | 익명화한 파티션 형식 예시 |
| 1세부 수신 항목 | 1세부 → KETI | tier1-submission.json | 형식 예시 |
| 계획안 (1·2회차) | 성균관대 → KETI | scheduler-plan.json | 형식 예시 |
| 결정 이력 · 승인 결과 | KETI → 성균관대, 화면 | scheduler-decision.json | 익명화한 승인·예약·해제 형식 예시 |
| 프로파일 결과 | 연세대 → KETI | profile-result.json | 형식 예시 |

로컬 운영 데이터를 수집하려면 대상 노드를 `KETI_EXAMPLE_NODE` 환경 변수로 지정하고 `python3 make_examples.py`를 실행합니다(queue-api 18090, scheduler 18092 포트 포워드 필요). 이 명령은 로컬 예제 파일을 실제 환경 데이터로 덮어씁니다. 생성 결과는 공개용 예제와 구분해 로컬에 보관하고, 공개 전 식별정보를 다시 검토해야 합니다.
