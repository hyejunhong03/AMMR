# Red Cube Pick 실험 기록

이 문서는 red cube pick baseline을 만들면서 겪은 주요 시행착오, 채택한
설정, 버린 방향, 핵심 산출물 경로를 한곳에 묶어 둔 작업 기록이다. Raw
dataset, checkpoint, rollout trace는 기존 위치에 그대로 두고, 이 문서는
그 파일들을 추적하기 위한 인덱스 역할을 한다.

현재 상태와 다음 연구 계획을 빠르게 보려면
`docs/project_status_summary_20261003.md`를 먼저 확인한다.

주의: episode 수집이 진행 중일 때는 active raw dataset directory를
이동하거나 삭제하지 않는다.

## 현재 상태

단일 red cube pick baseline은 아래 checkpoint에서 동결했다.

```text
outputs/train/phase2_red_cube_pick_only_lift060_h020_d8_sample60_ypos_hard_from_base_20000/checkpoints/015000/pretrained_model
```

이 baseline은 기존 local red-cube workspace에 대한 baseline이다. 새로
recentered table에서 정의한 safe placement workspace에 대해서는 raw
수집과 LeRobot 변환을 완료했고, 이제 기존 baseline에서 adaptation 학습을
진행하는 단계다.

현재 진행 중인 active experiment는 workspace 확장 실험이다.

```text
table footprint:       x=[0.095, 0.253], y=[-0.118, 0.118]
safe cube placement:   x=[0.130, 0.210], y=[-0.095, 0.095], z=0.115
task:                  pick the red cube
collection preset:     research_pick_only
```

safe placement 영역은 recentered table 전체를 10 mm grid로 훑고,
IK + vertical approach + 60 mm lift가 모두 되는 내부 영역만 골라서
정의했다.

```text
full table cube-center grid: 259/308 pass
safe placement grid:        180/180 pass
```

핵심 파일:

```text
outputs/eval/recentered_table_grid10mm_edges_ik_summary.json
outputs/eval/recentered_table_grid10mm_edges_ik_map.png
outputs/eval/recentered_table_safe_placement_grid10mm_positions.json
outputs/eval/recentered_table_safe_placement_grid10mm_h020_d8_poses.json
```

## 물리 스택에서 배운 점

초기 grasp instability는 주로 VLA 문제가 아니라 adaptive gripper의
mimic joint 구조를 잘못 drive한 문제였다.

채택한 물리 설정:

```text
gripper_controller만 drive
follower/mimic gripper joint는 직접 drive하지 않음
follower는 PhysX/Newton mimic constraint에 맡김
dynamic-contact expert 검증에서는 object dynamic + collision enabled
physical-contact expert 검증에서는 grasp assist off
stable cube grasp close command: -0.245
open command: 0.08
```

버렸거나 제한적으로만 쓰는 방향:

```text
drive_all_gripper_joints=True 계열은 wrist/gripper instability를 유발
close=-0.18은 dynamic contact lift에 파지력이 부족
gripper latch는 tail replay 일부에는 도움됐지만 full rollout에서는 성능 악화
GUI 수집/평가는 렌더링 부하 때문에 더 느린 arm speed가 필요할 수 있음
```

관련 문서:

```text
docs/dynamic_contact_grasp_mode.md
docs/physics_tuning_handoff_request_info.md
docs/physics_tuning_requested_materials_response.md
```

## 데이터 수집 방향 변화

과제는 pick-and-place에서 pick-only로 좁혔다. release/place까지 포함하면
파지 실패와 post-grasp handling 실패가 섞여서 원인 분석이 어려워졌기
때문이다.

현재 연구용 preset은 안정적인 lift와 hold를 요구한다.

```text
min_lift_m:   0.05
min_hold_sec: 2.0
close_hold:   0.8
lift_hold:    2.0
final_hold:   1.0
```

scripted expert는 hardcoded joint pose가 아니라 IK 기반이다.

```text
object position -> IK -> pose cache
vertical pre-approach
continuous descent
60 mm lift target
20 mm pre-approach height
8 descent intervals
```

주요 기존 dataset:

```text
data/lerobot/phase2_red_cube_pick_only_lift060_h020_d8_sample60_train
data/lerobot/phase2_red_cube_pick_only_lift060_h020_d8_sample60_ypos_train
data/lerobot/phase2_red_cube_pick_only_lift060_h020_d8_sample60_ypos_hard_train
```

현재 wide-workspace raw collection target:

```text
data/smolvla_raw_recentered_safe_x130_210_y095
```

이 dataset은 우선 기존 episode와 섞지 않는다. 먼저 새 workspace에서
새 데이터만으로, 또는 frozen baseline에서 새 데이터로 adaptation했을 때
성능이 어떻게 바뀌는지 분리해서 확인해야 한다.

수집 결과:

```text
raw dataset:       data/smolvla_raw_recentered_safe_x130_210_y095
episodes:          120
success:           120/120 = 100.0%
demo_exit_code:    0 for 120/120
failure_reason:    none for 120/120
object x range:    [0.130, 0.210]
object y range:    [-0.095, 0.095]
object z range:    [0.115, 0.115]
lift_delta_m:      min=0.0598, median=0.0605, max=0.0608
num_steps:         min=402, median=425, max=476
summary file:      outputs/eval/recentered_table_safe_raw_collection_summary.json
```

이 결과는 scripted expert가 recentered safe placement 영역에서 안정적으로
동작한다는 근거다. VLA rollout 성능은 아직 별도 학습과 held-out 평가가
필요하다.

LeRobot 변환 결과:

```text
lerobot dataset:    data/lerobot/recentered_safe_x130_210_y095_train
repo_id:            ammr/recentered_safe_x130_210_y095_train
episodes:           120
frames:             51354
fps:                10
image key:          observation.images.wrist
image shape:        256x256x3
state/action dim:   7 / 7
stats:              data/lerobot/recentered_safe_x130_210_y095_train/meta/stats.json
info:               data/lerobot/recentered_safe_x130_210_y095_train/meta/info.json
```

변환 명령:

```bash
.venv/bin/python scripts/convert_to_lerobot.py \
  --raw-dir data/smolvla_raw_recentered_safe_x130_210_y095 \
  --repo-id ammr/recentered_safe_x130_210_y095_train \
  --root data/lerobot/recentered_safe_x130_210_y095_train \
  --only-success
```

새 workspace 학습은 기존에 동결한 `ypos-hard 015000` baseline에서
adaptation한다. 주의할 점은 checkpoint의 `train_config.json` 안에
기록된 `policy.pretrained_path`가 아직 원래 SmolVLA base 경로라는 점이다.
따라서 학습 명령에서는 반드시 `--policy.pretrained_path`를 새로 override해서
동결 baseline checkpoint를 가리키게 한다.

학습 스크립트:

```text
scripts/run_recentered_safe_from_yposhard015k_train20k.sh
```

예상 출력:

```text
outputs/train/recentered_safe_x130_210_y095_from_yposhard015k_20000
checkpoints: 005000, 010000, 015000, 020000
```

학습 후 checkpoint 선택은 training 위치가 아니라 별도 held-out Set D에서
한다. Set D 위치 후보는 이미 생성해 두었다.

```text
positions:          outputs/eval/recentered_safe_setD_in_positions.json
count:              50
seed:               3001
safe region:        x=[0.130,0.210], y=[-0.095,0.095]
raw/grid exclusion: nearest xy distance >= 3 mm
output spacing:     >= 6 mm
planned pose cache: outputs/eval/recentered_safe_setD_in_h020_d8_poses.json
eval script:        scripts/run_recentered_safe_adapt_setD_eval.sh
```

Set D pose planning 결과:

```text
report:     outputs/eval/recentered_safe_setD_in_h020_d8_report.txt
status:     PLAN OK
solved:     50/50
pose cache: outputs/eval/recentered_safe_setD_in_h020_d8_poses.json
```

Set D rollout 요약을 처음 만들 때 `summarize_pick_eval.py`의 기본 support가
기존 60개 local workspace로 남아 있어서, 대부분의 recentered-safe 위치가
`out range`로 잘못 분류됐다. 보고용으로는 raw 120개 recentered-safe
collection을 train support로 지정한 아래 파일을 사용한다.

```text
summary:     outputs/eval/recentered_safe_setD_yposhardadapt_summary_rawtrain_support.json
failures:    outputs/eval/recentered_safe_setD_yposhardadapt_failures_rawtrain_support.csv
support:     data/smolvla_raw_recentered_safe_x130_210_y095
support box: x=[0.130,0.210], y=[-0.095,0.095]
```

Set D 결과:

```text
005000: raw=19/50, clean=13/50, violations=9/50
010000: raw=16/50, clean=15/50, violations=2/50
015000: raw=30/50, clean=27/50, violations=6/50
020000: raw=36/50, clean=36/50, violations=1/50
```

현재 best는 20k checkpoint지만 gate는 아직 통과하지 못했다.

```text
best checkpoint: outputs/train/recentered_safe_x130_210_y095_from_yposhard015k_20000/checkpoints/020000/pretrained_model
clean success:   36/50 = 72.0%
velocity rate:   1/50 = 2.0%
gate:            clean >= 80%, spike <= 5%
status:          not passed
```

20k 실패 14건은 모두 approach miss로 분류된다. 이 중 11건은 `x < 0.17`
저 x 쪽에 있고, 6건은 `|y| >= 0.07` edge에 있다. 따라서 다음 개선은
전 영역을 균등하게 늘리기보다 low-x 쪽과 high-|y| edge를 겨냥한 correction
data를 추가하는 방향이 우선이다.

다만 20k까지 성능이 계속 상승하고 있었기 때문에, 데이터 추가 전에 40k까지
이어 학습을 먼저 확인했다. Set D 결과는 아래와 같다.

```text
030000: raw=45/50, clean=44/50, violations=1/50
040000: raw=46/50, clean=45/50, violations=2/50
```

30k와 40k 모두 recentered safe workspace gate를 통과했다.

```text
gate: clean >= 80%, spike <= 5%
030000: clean 88.0%, violation 2.0%
040000: clean 90.0%, violation 4.0%
```

현재 best는 40k checkpoint다.

```text
outputs/train/recentered_safe_x130_210_y095_from_yposhard015k_20000/checkpoints/040000/pretrained_model
```

다만 Set D는 checkpoint 선택에 사용했으므로, recentered-safe baseline을
최종 보고 수치로 고정하려면 새 held-out Set E에서 한 번 더 확인해야 한다.

Set E 최종 held-out 검증도 완료했다. Set E는 40k checkpoint를 고정한 뒤
확인한 별도 held-out set이다.

```text
checkpoint: outputs/train/recentered_safe_x130_210_y095_from_yposhard015k_20000/checkpoints/040000/pretrained_model
positions:  outputs/eval/recentered_safe_setE_in_positions.json
poses:      outputs/eval/recentered_safe_setE_in_h020_d8_poses.json
report:     outputs/eval/recentered_safe_setE_in_h020_d8_report.txt
rollout:    outputs/eval/recentered_safe_setE_yposhardadapt_040000_snap_thr004_x60.json
summary:    outputs/eval/recentered_safe_setE_yposhardadapt_040000_summary_rawtrain_support.json
failures:   outputs/eval/recentered_safe_setE_yposhardadapt_040000_failures_rawtrain_support.csv
```

Set E 결과:

```text
raw success:          52/60 = 86.7%
clean success:        51/60 = 85.0%
velocity violations:   1/60 = 1.7%
clean Wilson 95% CI:  [73.9%, 91.9%]
failure classes:      approach_miss 8, success_with_state_jump 1
```

Set E에서도 gate를 통과했다.

```text
gate:   clean >= 80%, velocity violation <= 5%
status: passed
```

따라서 recentered safe workspace의 단일 red cube pick baseline은 40k
checkpoint로 동결해도 된다. 보고할 때 대표 수치는 raw 52/60이 아니라
clean 51/60을 사용한다.

40k에서 raw success 46/50과 clean success 45/50이 다른 이유는 episode
`#011` 때문이다.

```text
episode:                 #011
object xy:               (0.1884, -0.0508)
success:                 true
velocity_ok:             false
failure class:           success_with_stuck
velocity violation type: sim_state_spike
stuck_event:             true
state_jump_event:        true
tcp_min:                 21.6 mm
lift_m:                  24 mm
peak joint:              joint6
state jump:              1.959 rad
action step at peak:     0.013 rad
max joint velocity:      3.26 rad/s
```

이 케이스는 모델 command 자체가 크게 튄 것이 아니라, action은 부드러운데
sim state가 크게 튄 경우다. 물체는 lift되어 raw success로 잡혔지만,
`tcp_min`이 2 cm 이상이고 stuck/state-jump가 같이 발생했기 때문에 안정적인
중앙 파지라기보다 비대칭 접촉 또는 끼임이 동반된 성공으로 본다. 실제 로봇
이관 기준에서는 위험한 충돌성 동작이므로 clean success에서 제외한다.

## 평가에서 배운 점

주요 지표는 raw success가 아니라 clean success다.

```text
clean_success = success == true and velocity_ok == true
```

raw success만 보면 contact-induced state jump가 있는 episode도 성공으로
보일 수 있다. 실제 로봇 이관을 생각하면 이런 episode는 clean pick으로
세면 안 된다.

red-cube single-object rollout 기본 설정:

```text
--gripper-snap
--gripper-snap-threshold 0.04
--max-action-step 0.05
--max-joint-velocity 0.5
--n-action-steps 50
--noise-mode fixed-random
--noise-seed 0
```

gripper latch는 default evaluation setting으로 쓰지 않는다. 이전 실험에서
full rollout 성능이 나빠졌다.

## 동결한 baseline

가장 좋은 동결 checkpoint:

```text
outputs/train/phase2_red_cube_pick_only_lift060_h020_d8_sample60_ypos_hard_from_base_20000/checkpoints/015000/pretrained_model
```

manifest:

```text
outputs/eval/red_cube_baseline_yposhard015k_manifest.json
```

최종 held-out reporting set:

```text
outputs/eval/setC_in_general_yposhard015k_snap_thr004_x47.json
outputs/eval/setC_in_ypos_high_spaced_yposhard015k_snap_thr004_x22.json
outputs/eval/setC_yposhard015k_summary.json
outputs/eval/setC_yposhard015k_failures.csv
```

table 영역을 recentered table로 바꾸기 전, 기존 local workspace에서의
primary held-out 결과는 아래와 같다. 이 숫자가 현재 단일 red cube pick
baseline의 핵심 보고 수치다.

```text
Primary set C general:
  raw success:       47/47 = 100.0%
  clean success:     46/47 = 97.9%
  velocity violation: 1/47 = 2.1%
  failure class:     success_with_state_jump 1건

Set C ypos-high spaced:
  raw success:       20/22 = 90.9%
  clean success:     20/22 = 90.9%
  velocity violation: 0/22 = 0.0%
  failure class:     approach_miss 2건
```

최종 요약:

```text
setC general:      raw=47/47, clean=46/47, violations=1/47
setC ypos-high:    raw=20/22, clean=20/22, violations=0/22
```

이 결과는 “검증된 기존 workspace 안에서 단일 red cube를 높은 성공률로
pick할 수 있다”는 주장까지 뒷받침한다. table 전체 generalization이나
language grounding까지 뒷받침하는 결과는 아니다.

## Recentered Table 실험을 시작한 이유

기존 학습 workspace는 사용자가 화면에서 보는 table보다 훨씬 작았다.
myCobot 280의 nominal reach radius 안에 table corner가 들어온다고 해서,
그 지점이 실제 grasp orientation, vertical approach, 60 mm lift까지 모두
만족한다는 뜻은 아니다.

recentered table은 table을 로봇 쪽으로 당겼지만, table 전체가 여전히
valid pick workspace는 아니었다.

```text
full cube-center grid: 259/308 pass
failures:
  no_lift_solution:                    13
  ik_no_vertical_approach_solution:    32
  ik_no_grasp_solution_beyond_reach:    4
```

그래서 all-pass interior만 safe placement 영역으로 정의했다.

```text
x=[0.130, 0.210]
y=[-0.095, 0.095]
z=0.115
```

관련 코드 상수:

```text
scripts/ammr_mycobot_interface.py
  TASK_TABLE_X_RANGE = (0.095, 0.253)
  TASK_TABLE_Y_RANGE = (-0.118, 0.118)
  TASK_PLACEMENT_X_RANGE = (0.130, 0.210)
  TASK_PLACEMENT_Y_RANGE = (-0.095, 0.095)
```

`scripts/plan_grasp_poses.py`의 random sampler는 이제 table footprint가
아니라 `TASK_PLACEMENT_*_RANGE`를 사용한다.

## 현재 수집 명령

Isaac Sim이 이미 실행 중이고 ROS2 bridge가 살아 있는 상태에서 실행한다.

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

.venv/bin/python scripts/collect_episodes.py \
  --count 120 \
  --poses-file outputs/eval/recentered_table_safe_placement_grid10mm_h020_d8_poses.json \
  --shuffle \
  --seed 0 \
  --object red_cube \
  --park-non-targets \
  --instruction "pick the red cube" \
  --task-name red_cube_recentered_safe_pick_only \
  --output-dir data/smolvla_raw_recentered_safe_x130_210_y095 \
  --collection-preset research_pick_only \
  --close -0.245 \
  --arm-speed 0.12 \
  --state-timeout 18
```

수집이 끝나면 변환/학습 전에 raw metadata부터 확인한다.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("data/smolvla_raw_recentered_safe_x130_210_y095")
eps = sorted(root.glob("episode_*"))
ok = 0
for ep in eps:
    meta = json.loads((ep / "metadata.json").read_text())
    ok += bool(meta.get("success"))
print(f"episodes={len(eps)} success={ok}/{len(eps)}")
PY
```

## 다음 평가 gate

recentered safe workspace 결과는 training-position success로 최종 보고하지
않는다. 별도 held-out safe placement set을 만들어 평가해야 한다.

권장 gate:

```text
safe held-out clean success >= 80%
velocity/spike rate <= 5%
failure table에서 approach miss와 state jump를 분리해서 기록
```

이 gate는 제품 수준의 최종 성능 기준이나 100% 성공 요구가 아니다. 다음
연구 단계로 넘어가도 되는지를 판단하기 위한 engineering threshold다.

`clean >= 80%`는 단일 물체 pick 파이프라인이 대체로 작동한다는 최소선이다.
60~70% 수준에서는 이후 two-object 실험에서 실패 원인이 grasp 실패인지
language grounding 실패인지 분리하기 어렵다. 반대로 90~95% 이상을 요구하면
red cube tuning에 지나치게 묶여 장기 목표인 language grounding과 unseen
object 실험으로 넘어가기 어렵다.

`velocity/spike <= 5%`는 실제 로봇 이관 관점의 안정성 최소선이다. raw
success가 높아도 state jump, stuck, sim_state_spike가 자주 발생하면 안정적
성공으로 보면 안 된다. 5% 이하는 위험한 contact/state jump가 예외적으로
발생하는 수준으로 보고, 10~15% 이상이면 아직 안정화가 부족한 상태로 본다.

즉 이 gate의 의미는 다음과 같다.

```text
clean >= 80%:
  grasp baseline으로 쓸 만큼 충분히 성공한다

velocity/spike <= 5%:
  위험한 contact/state jump가 드물다
```

이 gate를 통과하면 다음 과학적 단계는 two-object language grounding이다.

```text
red cube + blue cylinder를 같은 scene에 배치
target instruction을 균형 있게 구성
target-selection accuracy와 grasp success를 분리 측정
language ablation 포함
```

unseen object 주장은 seen object에 대한 language grounding이 먼저 된 뒤에
시작해야 한다.
