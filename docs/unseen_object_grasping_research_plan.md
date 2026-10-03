# AMMR SmolVLA Unseen Object Grasping Research Plan

현재 red cube baseline, recentered table 결과, VLA/VLM+IK 구조 논의,
그리고 다음 단계의 요약은 `docs/project_status_summary_20261003.md`에
정리되어 있다.

## 연구 목표

최종 목표는 AMMR 환경에서 wrist RGB camera, 자연어 명령, robot state를 입력으로 받아 학습 때 보지 못한 물체도 안정적으로 pick할 수 있는 SmolVLA 기반 manipulation policy를 만드는 것이다.

핵심 연구 질문은 다음과 같다.

> SmolVLA가 myCobot embodiment에서 특정 물체를 외운 것이 아니라, 새로운 물체의 grasp affordance를 학습했는가?

따라서 red cube나 blue cylinder의 개별 성공률을 높이는 것이 최종 목표가 아니다. 이들은 시스템 검증용 primitive task로만 사용한다.

## 기존 Red / Blue Task의 역할

Red cube와 blue cylinder는 앞으로 메인 연구 대상이 아니라 regression test로 사용한다.

- Red cube: 로봇, 그리퍼, rollout 파이프라인이 망가지지 않았는지 확인하는 기본 회귀 테스트
- Blue cylinder: 단순 shape variation sanity check
- 최종 unseen object 성능 주장에는 사용하지 않음
- 논문에서는 초기 시스템 검증용 primitive task로 설명

Red / blue 성공률을 계속 끌어올리는 데 과도하게 시간을 쓰면 최종 목표인 unseen object grasping과 어긋날 수 있다.

## Phase 1: Single Unseen Object Grasping

첫 번째 연구 단계는 테이블 위에 물체가 하나 있을 때 처음 보는 물체도 pick할 수 있는지 확인하는 것이다.

목표:

> 테이블 위에 물체가 하나 있을 때, `"pick the object"` 명령으로 학습에 없던 물체를 잡을 수 있는가?

이 단계에서는 target selection 문제가 없기 때문에 언어 grounding보다 grasp affordance 일반화를 평가한다.

데이터 구성:

- Train objects: 여러 모양, 크기, 색상의 물체
- Test objects: 학습에 절대 포함하지 않은 새로운 물체
- Instruction: `"pick the object"` 또는 `"pick up the object"`로 단순화
- Randomization: 위치, 회전, 조명, 배경, 물체 크기

평가:

- Seen object success
- Unseen object success
- Failure reason: no reach, no attach, slipped, unstable action

이 단계가 성공해야 특정 물체를 외운 것이 아니라 물체를 잡는 행동을 학습했다고 주장할 수 있다.

### Phase 1-0: Zero-Shot Baseline

Fine-tuning 전에 zero-shot baseline을 먼저 측정한다.

목표:

> SmolVLA pretrained base weight가 AMMR/myCobot 환경에 추가 gradient update 없이 얼마나 전이되는가?

주의할 점:

- 기존 d6/d7 checkpoint는 fine-tuned 모델이므로 zero-shot이 아니다.
- 공식 `lerobot/smolvla_base`는 3 camera, 6-D state/action 구조라 AMMR에 바로 연결할 수 없다.
- 따라서 baseline은 공식 base weight를 그대로 쓰되, AMMR wrist camera와 7-D state/action adapter를 붙인 `models/smolvla_base_ammr_zeroshot`으로 실행한다.
- 이 baseline은 **zero-gradient AMMR-adapted SmolVLA base**로 표기한다.

검증 순서:

1. `scripts/prepare_smolvla_zeroshot_checkpoint.py`로 AMMR-compatible checkpoint 생성
2. `scripts/check_policy_outputs.py`로 action shape, joint limit, gripper range 확인
3. Isaac rollout으로 single-object baseline 측정
4. 이후 `d8_unseen_single_object` fine-tuning 결과와 비교

## Phase 2: Multi-Object Language-Conditioned Picking

두 번째 단계는 여러 물체가 있을 때 자연어로 지시한 물체를 고르고 pick하는 것이다.

예시 명령:

- `"pick the blue object"`
- `"pick the cylinder"`
- `"pick the object on the left"`
- `"pick the taller object"`

이 단계에서는 두 능력을 분리해서 평가해야 한다.

- Language grounding: 지시한 target object를 고르는 능력
- Manipulation: 선택한 물체를 실제로 grasp하는 능력

Phase 1이 안정되기 전에 바로 multi-object task로 가면 실패 원인이 target selection인지 grasp 실패인지 분리하기 어렵다.

## Phase 3: AMMR Full Task

마지막 단계는 AMMR 전체 task로 확장한다.

목표 시나리오:

1. 테이블 앞으로 이동
2. 목표 물체 확인
3. 팔이 닿는지 검사
4. 안 닿으면 base alignment
5. Pick
6. Place

현실적인 구조는 mobile base와 manipulation을 완전 end-to-end로 바로 통합하지 않고 역할을 나누는 것이다.

Mobile base:

- Mapping
- Localization
- Waypoint navigation
- Base alignment

SmolVLA:

- Wrist image + language + arm state 입력
- Arm joint target action 출력
- Pick / place manipulation 담당

## 지금부터의 우선순위

1. Cleanup manifest를 기준으로 기존 primitive 실험 산출물을 정리한다.
2. Red cube 안정화에 더 매몰되지 않는다.
3. 기존 d6 red baseline은 보존한다.
4. Red / blue mixed fine-tuning 결과는 참고용으로만 유지한다.
5. Zero-shot baseline을 먼저 측정한다.
6. Unseen object grasping용 dataset schema를 새로 정의한다.
7. Single-object grasp dataset부터 만든다.
8. Train/test split은 반드시 object identity 기준으로 분리한다.
9. SmolVLA fine-tuning 목표를 red/blue 분류가 아니라 grasp 일반화로 재설계한다.
10. 평가는 zero-shot, seen object success, unseen object success를 분리해서 보고한다.

## 2026-09-22 진행 상태

우선 1~4번 기반 작업을 진행했다.

1. Zero-shot 결과 정리
   - 결과 문서: `docs/zero_shot_transfer_results_20260922.md`
   - 공식 baseline은 `models/smolvla_base_ammr_zeroshot_camera2_wrist`
   - 결과: red cube 0/10, velocity 10/10 통과
   - 결론: camera2 wrist mapping으로 action 안정성은 확보됐지만, pretrained base는 AMMR grasp를 zero-shot으로 수행하지 못했다.
2. Unseen object catalog 생성
   - catalog 파일: `data/unseen_single_object_catalog_v0.json`
   - 설명 문서: `docs/unseen_single_object_catalog_v0.md`
   - train/test object identity를 분리했다.
3. Simulator object 구조 확장
   - active red/blue object 정의를 `scripts/ammr_mycobot_interface.py`의 catalog로 중앙화했다.
   - Isaac scene 생성, reset, object state publish, grasp assist가 active object list를 loop로 처리한다.
   - 기존 `object_state.npy` shape를 유지하기 위해 true unseen object는 아직 active runtime schema에 넣지 않았다.
4. Single-object pose generation 준비
   - `scripts/plan_grasp_poses.py`가 object metadata를 pose cache에 기록한다.
   - `scripts/collect_episodes.py`가 pose-cache metadata를 episode metadata로 전달한다.
   - evaluate/rollout/scripted demo의 object-state index 접근을 공통 helper로 통일했다.

## Unseen Object Dataset에 필요한 Metadata

각 episode에는 최소한 다음 정보를 기록한다.

- `object_id`
- `seen_split`: `train` 또는 `test`
- `shape_class`
- `size`
- `color`
- `object_pose`
- `instruction`
- `success`
- `failure_reason`
- `policy_input`: privileged object state는 학습 입력에서 제외

Object state는 시뮬레이터 평가와 분석에는 사용할 수 있지만 policy input에는 넣지 않는다.

## 1차 논문 목표

석사 논문 기준의 현실적인 1차 목표는 다음과 같이 잡는다.

> AMMR arm-only setup에서 SmolVLA를 파인튜닝하여 다양한 물체에 대한 single-object pick을 학습하고, 학습에 사용하지 않은 물체에 대한 일반화 성능을 평가한다.

성공 시 확장 목표:

> 복수 물체 환경에서 자연어 조건에 따라 target object를 선택하고 pick한다.

## 결론

앞으로의 연구 방향은 red cube / blue cylinder 성공률 개선이 아니라 object diversity 기반 generalization 실험이다.

Red / blue task는 시스템이 망가지지 않았는지 확인하는 회귀 테스트로 유지하고, 메인 실험은 unseen object grasping으로 이동한다.
