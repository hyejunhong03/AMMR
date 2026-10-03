# AMMR SmolVLA Red Cube Baseline 및 Unseen Grasping 전환 요약

작성일: 2026-10-03

이 문서는 지금까지 진행한 red cube pick 실험, recentered table 실험,
VLA/VLM+IK 구조 논의, 그리고 unseen object grasping으로 넘어가기 위한
다음 계획을 한곳에 정리한 현재 상태 보고서다. 세부 실험 로그는
`docs/red_cube_pick_experiment_log.md`에 있고, 이 문서는 의사결정과
보고용 요약을 목적으로 한다.

## 한 줄 결론

단일 red cube pick은 recentered safe workspace에서 held-out clean
success 51/60 = 85.0%, velocity violation 1/60 = 1.7%까지 확인했다.
따라서 red cube single-object baseline은 회귀 테스트로 동결하고, 다음
연구 단계는 두 물체 환경에서 언어 지시로 target을 고르는 language
grounding 검증이다.

## 현재 목표

장기 목표는 학습 때 보지 못한 물체를 자연어 지시에 따라 안정적으로
파지하는 것이다. 현재까지의 red cube 실험은 최종 목표 자체가 아니라
다음 조건을 만족하는지 확인하기 위한 baseline 단계였다.

- myCobot 280 + adaptive gripper 물리 스택이 안정적인가
- wrist RGB camera + language + robot state로 SmolVLA rollout이 되는가
- 단일 물체 pick이 scripted expert와 VLA rollout 모두에서 재현되는가
- 실패를 raw success가 아니라 clean success 기준으로 분석할 수 있는가

이제 red cube는 메인 연구 대상이 아니라 이후 실험의 회귀 테스트로
사용한다.

## 시스템 구성

현재 기본 설정은 아래와 같다.

```text
simulator:        Isaac Sim
robot:            myCobot 280 M5 + adaptive gripper
camera:           wrist RGB camera, 256x256
model:            SmolVLA fine-tuning
input:            wrist image + language text + robot state 7D
output:           absolute joint target 7D [joint1..joint6, gripper]
gripper open:     0.08
gripper close:   -0.245
task string:      pick the red cube
```

대표 rollout 설정:

```text
--gripper-snap
--gripper-snap-threshold 0.04
--max-action-step 0.05
--max-joint-velocity 0.5
--n-action-steps 50
--noise-mode fixed-random
--noise-seed 0
```

## 물리 안정화에서 얻은 결론

초기 불안정성의 핵심 원인은 VLA가 아니라 adaptive gripper mimic joint
구조와 drive 방식의 충돌이었다.

채택한 설정:

- `gripper_controller` 1개만 직접 drive한다.
- follower/mimic gripper joint는 직접 drive하지 않는다.
- follower는 PhysX/Newton mimic constraint에 맡긴다.
- object는 dynamic + collision enabled 상태에서 평가한다.
- grasp assist는 baseline 평가에서 끈다.
- close command는 sweep 결과 `-0.245`를 사용한다.

버린 방향:

- `drive_all_gripper_joints=True` 계열은 gripper/wrist instability를
  유발했다.
- close `-0.18`은 dynamic contact lift에 파지력이 부족했다.
- `--gripper-latch-after-close`는 tail replay에는 도움이 되는 경우가
  있었지만 full rollout에서는 성능을 떨어뜨렸다.

## 평가 기준

대표 지표는 raw success가 아니라 clean success다.

```text
clean_success = success == true and velocity_ok == true
```

이유는 물체가 들어올려졌더라도 contact-induced state jump, stuck event,
sim_state_spike가 있으면 실제 로봇 이관 기준으로는 안정적인 성공이
아니기 때문이다.

현재 gate:

```text
held-out clean success >= 80%
velocity violation rate <= 5%
```

## 기존 Local Red Cube Baseline

table 영역을 recenter하기 전의 local workspace에서 아래 checkpoint를
동결했다.

```text
checkpoint:
outputs/train/phase2_red_cube_pick_only_lift060_h020_d8_sample60_ypos_hard_from_base_20000/checkpoints/015000/pretrained_model

manifest:
outputs/eval/red_cube_baseline_yposhard015k_manifest.json
```

주요 held-out 결과:

```text
Set C general:
  raw success:        47/47 = 100.0%
  clean success:      46/47 = 97.9%
  velocity violation:  1/47 = 2.1%

Set C ypos-high spaced:
  raw success:        20/22 = 90.9%
  clean success:      20/22 = 90.9%
  velocity violation:  0/22 = 0.0%
```

이 결과가 뒷받침하는 주장은 제한적이다.

- 말할 수 있음: 검증된 기존 workspace 안에서 단일 red cube는 높은
  성공률로 pick할 수 있다.
- 말할 수 없음: table 전체에서 가능하다, 언어 grounding을 한다,
  unseen object를 잡을 수 있다.

## Table Recenter 및 Safe Placement Workspace

기존 workspace는 화면에서 보이는 table 전체에 비해 매우 작았다. myCobot
280의 nominal reach radius 안에 어떤 지점이 들어온다고 해서 그 지점에서
vertical approach, stable grasp, 60 mm lift가 모두 가능한 것은 아니다.

그래서 table을 로봇 쪽으로 recenter하고, table 전체를 10 mm grid로
검증했다.

```text
table footprint:
  x=[0.095, 0.253]
  y=[-0.118, 0.118]

full table grid:
  259/308 pass

safe cube placement:
  x=[0.130, 0.210]
  y=[-0.095, 0.095]
  z=0.115

safe placement grid:
  180/180 pass
```

즉, table 전체가 아니라 IK + vertical approach + lift가 모두 되는 내부
영역을 cube placement 영역으로 제한했다.

관련 파일:

```text
outputs/eval/recentered_table_grid10mm_edges_ik_summary.json
outputs/eval/recentered_table_grid10mm_edges_ik_map.png
outputs/eval/recentered_table_safe_placement_grid10mm_positions.json
outputs/eval/recentered_table_safe_placement_grid10mm_h020_d8_poses.json
outputs/eval/recentered_table_safe_placement_manifest.json
```

## Recentered Safe Dataset

새 workspace에서 scripted expert로 120 episode를 수집했다.

```text
raw dataset:
data/smolvla_raw_recentered_safe_x130_210_y095

episodes:
120

scripted success:
120/120

object range:
x=[0.130, 0.210]
y=[-0.095, 0.095]
z=0.115
```

LeRobot 변환 결과:

```text
lerobot dataset:
data/lerobot/recentered_safe_x130_210_y095_train

repo_id:
ammr/recentered_safe_x130_210_y095_train

episodes:
120

frames:
51354

stats:
data/lerobot/recentered_safe_x130_210_y095_train/meta/stats.json
```

중요한 구분:

- LeRobot 변환은 fine-tuning이 아니다.
- fine-tuning은 `outputs/train/...` 아래 checkpoint를 만드는 과정이다.
- recentered safe fine-tuning은 기존 local baseline checkpoint에서 시작했지만,
  dataloader에는 새 120 episode만 사용했다.
- 따라서 기존 데이터가 batch에 섞인 것은 아니고, 기존 지식은 초기 weight에
  남아 있는 형태다.

## Recentered Safe Fine-Tuning 및 Set D

학습 시작점:

```text
base checkpoint:
outputs/train/phase2_red_cube_pick_only_lift060_h020_d8_sample60_ypos_hard_from_base_20000/checkpoints/015000/pretrained_model
```

새 학습 출력:

```text
outputs/train/recentered_safe_x130_210_y095_from_yposhard015k_20000
```

Set D는 checkpoint 선택용 held-out set이다.

```text
positions:
outputs/eval/recentered_safe_setD_in_positions.json

poses:
outputs/eval/recentered_safe_setD_in_h020_d8_poses.json

count:
50

condition:
safe workspace 안쪽, raw training positions와 Set D 내부 위치 간 최소 간격 적용
```

Set D 결과:

```text
005000: raw=19/50, clean=13/50, violations=9/50
010000: raw=16/50, clean=15/50, violations=2/50
015000: raw=30/50, clean=27/50, violations=6/50
020000: raw=36/50, clean=36/50, violations=1/50
030000: raw=45/50, clean=44/50, violations=1/50
040000: raw=46/50, clean=45/50, violations=2/50
```

Set D 기준으로는 40k가 가장 높았지만, Set D는 checkpoint 선택에 사용했으므로
최종 보고 숫자는 새 held-out Set E에서 확인해야 했다.

## Final Held-Out Set E

Set E는 40k checkpoint를 고정한 뒤 새로 평가한 최종 held-out set이다.

```text
checkpoint:
outputs/train/recentered_safe_x130_210_y095_from_yposhard015k_20000/checkpoints/040000/pretrained_model

rollout:
outputs/eval/recentered_safe_setE_yposhardadapt_040000_snap_thr004_x60.json

summary:
outputs/eval/recentered_safe_setE_yposhardadapt_040000_summary_rawtrain_support.json

failure csv:
outputs/eval/recentered_safe_setE_yposhardadapt_040000_failures_rawtrain_support.csv
```

Set E 결과:

```text
raw success:          52/60 = 86.7%
clean success:        51/60 = 85.0%
velocity violations:   1/60 = 1.7%
clean Wilson 95% CI:  [73.9%, 91.9%]
failure classes:      approach_miss 8, success_with_state_jump 1
```

결론:

- recentered safe workspace에서 gate를 통과했다.
- 대표 보고 숫자는 raw 52/60이 아니라 clean 51/60이다.
- 100%가 아니어도 baseline으로는 충분하다. 이후 실험에서는 이 baseline을
  regression test로 사용하면 된다.

동결할 현재 red cube checkpoint:

```text
outputs/train/recentered_safe_x130_210_y095_from_yposhard015k_20000/checkpoints/040000/pretrained_model
```

## 남은 실패의 해석

Set E의 non-clean 9건은 approach miss 8건과 success_with_state_jump 1건이다.

패턴:

- x가 낮은 쪽에서 실패가 상대적으로 많다.
- y<0 쪽 실패가 많다.
- 일부 |y| edge에서도 miss가 있다.
- clean하지 않은 성공은 물체를 들어올렸지만 state jump가 있어 실제 로봇
  기준으로 제외했다.

이것은 물리 스택이 완전히 망가졌다는 신호라기보다, policy 접근 정밀도와
contact configuration이 아직 완벽하지 않다는 신호다. 물리 문제는 초기보다
크게 정리됐지만, 접촉이 비대칭이거나 끼임이 생기면 state jump가 여전히
발생할 수 있다.

## 지금 주장할 수 있는 것과 없는 것

주장할 수 있는 것:

- scripted expert는 recentered safe workspace에서 120/120 성공했다.
- SmolVLA fine-tuned policy는 recentered safe held-out Set E에서 clean
  51/60을 달성했다.
- 단일 red cube pick은 regression baseline으로 사용할 수 있는 수준이다.
- clean success와 velocity violation을 분리해서 평가하는 프로토콜이 있다.

아직 주장하면 안 되는 것:

- table 전체에서 red cube를 안정적으로 잡는다.
- 언어 지시를 실제로 이해한다.
- unseen object를 잡을 수 있다.
- end-to-end VLA만으로 실제 로봇 안정성까지 해결됐다.

## VLA, VLM+IK, Modular VLA에 대한 현재 판단

현재 구현은 다음 형태의 end-to-end joint-action VLA baseline이다.

```text
input:
  wrist image + language + robot state

output:
  absolute joint target [joint1..joint6, gripper]
```

즉, 모델이 joint target 자체를 학습하고 있다. 이는 로봇 VLA에서 널리 쓰이는
방식 중 하나다. 다만 unseen object와 실제 로봇 안정성까지 생각하면,
장기적으로는 역할을 나누는 구조가 더 현실적일 수 있다.

권장 구조:

```text
vision-language model / VLA:
  무엇을 잡을지
  어디를 잡을지
  어떤 grasp pose 또는 skill parameter를 쓸지

robot controller / IK / motion planner:
  팔을 어떻게 움직일지
  joint trajectory를 어떻게 안정적으로 만들지
  joint limit, velocity, collision을 어떻게 만족할지
```

용어 정리:

- 모델이 바로 joint action을 내면 현재처럼 end-to-end VLA라고 부를 수 있다.
- 모델이 mask, point, pose, waypoint, skill parameter를 내고 IK/controller가
  움직임을 만들면 strict한 의미의 end-to-end VLA는 아니다.
- 다만 언어와 비전이 manipulation action의 핵심 결정을 하고, controller가
  안정적인 실행을 맡는다면 modular VLA-style manipulation system이라고
  설명할 수 있다.
- 논문에서는 과장하지 않고, 실제 출력과 controller 역할을 명확히 쓰는 것이
  중요하다.

## 관련 연구 흐름

모듈형 방향을 뒷받침하는 대표 흐름:

- SayCan: language model이 가능한 skill을 선택하고 low-level skill이 실행
- Code as Policies: language model이 robot control code 또는 policy를 구성
- VoxPoser: language/vision 기반 affordance map과 motion planning 결합
- MOKA, RoboPoint, OK-Robot: VLM이 target, point, affordance, skill 선택에 관여
- AnyGrasp, CLIPort: perception/affordance와 manipulation policy를 결합하는 방향

end-to-end VLA 방향의 대표 흐름:

- RT-1
- RT-2
- Open X-Embodiment / RT-X
- OpenVLA
- Octo
- pi0
- RDT-1B
- SmolVLA

핵심 해석:

- end-to-end VLA로 unseen object/generalization을 노리는 연구는 분명히 있다.
- 하지만 대체로 매우 큰 사전학습 데이터, 다양한 embodiment, 대규모 task
  coverage에 의존한다.
- myCobot + 단일 카메라 + 작은 수집 데이터에서는 pure end-to-end만으로
  unseen object까지 바로 가는 것보다, target/grasp decision과 controller를
  나누는 구조가 실험적으로 더 안정적일 가능성이 높다.

## 다음 단계

### 1. Red Cube Baseline 동결

현재 recentered safe checkpoint를 동결한다.

```text
outputs/train/recentered_safe_x130_210_y095_from_yposhard015k_20000/checkpoints/040000/pretrained_model
```

향후 새로운 데이터나 구조를 실험할 때마다 아래 회귀 테스트를 유지한다.

```text
single red cube, recentered safe workspace:
  target clean success >= 80%
  velocity violation <= 5%
```

### 2. Two-Object Language Grounding

다음 과학적 단계는 두 물체를 동시에 놓고 언어 지시로 target을 고르는지
검증하는 것이다.

권장 scene:

```text
objects:
  red cube
  blue cylinder

instructions:
  pick the red cube
  pick the blue cylinder
```

평가 지표를 반드시 분리한다.

```text
target selection accuracy:
  맞는 물체 쪽으로 접근했는가

grasp success:
  선택한 물체를 실제로 안정적으로 잡았는가

clean success:
  success and velocity_ok
```

좌우 배치와 위치 분포를 균형 있게 만들어야 한다. 그렇지 않으면 모델이
언어를 쓰지 않고 항상 왼쪽, 항상 가까운 물체 같은 shortcut을 배울 수 있다.

언어 ablation도 필요하다.

```text
instruction swap:
  red 지시와 blue 지시를 바꿨을 때 접근 대상이 바뀌는가

empty / neutral instruction:
  지시문을 제거하면 target selection이 50% 근처로 떨어지는가
```

### 3. Unseen Object로 확장

두 물체 language grounding이 확인된 뒤 unseen object 단계로 간다.

첫 단계는 너무 크게 잡지 않는다.

```text
train:
  red cube
  blue cylinder
  추가 seen objects

test:
  학습에 없던 색-형상 조합
  예: blue cube, red cylinder

longer-term test:
  학습에 없던 새로운 shape/object identity
```

unseen 주장은 object identity 기준 split이 있어야 한다. 같은 물체의 위치만
바뀐 것은 unseen object가 아니다.

## 문서 및 산출물 인덱스

상세 실험 로그:

```text
docs/red_cube_pick_experiment_log.md
```

unseen object 연구 계획:

```text
docs/unseen_object_grasping_research_plan.md
```

기존 local red cube baseline manifest:

```text
outputs/eval/red_cube_baseline_yposhard015k_manifest.json
```

recentered safe workspace manifest:

```text
outputs/eval/recentered_table_safe_placement_manifest.json
```

현재 recentered safe 최종 평가:

```text
outputs/eval/recentered_safe_setE_yposhardadapt_040000_summary_rawtrain_support.json
outputs/eval/recentered_safe_setE_yposhardadapt_040000_failures_rawtrain_support.csv
```

현재 가장 중요한 checkpoint:

```text
outputs/train/recentered_safe_x130_210_y095_from_yposhard015k_20000/checkpoints/040000/pretrained_model
```

## 최종 판단

현재 프로젝트는 단일 red cube pick baseline 완성 단계는 통과했다. 완성의
의미는 table 전체 또는 unseen object가 아니라, 정의된 recentered safe
workspace 안에서 clean success 기준으로 회귀 테스트에 쓸 수 있는 baseline을
확보했다는 뜻이다.

다음 연구 가치는 red cube 성공률을 100%에 가깝게 더 끌어올리는 것보다,
언어 지시가 실제 target selection에 영향을 주는지 확인하는 데 있다. 그
다음에 object diversity와 unseen object split으로 확장하는 것이 현재
목표에 가장 맞는 순서다.
