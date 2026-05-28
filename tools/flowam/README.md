# FloWAM Filtered Flow Conversion

This converter prepares the filtered FloWAM flow episodes for the local
PointWorld action-finetuning path.

Input flow data:

```text
/work/runyi_yang/FloWAM/data/FloWAM/flow_data_filtered
```

Original action data:

```text
/work/runyi_yang/FloWAM/data/FloWAM/origin
```

The WDS action layout is 54-D:

```text
left_arm(7), right_arm(7), left_hand(20), right_hand(20)
```

Run from the PointWorld repo root:

```bash
bash scripts/flowam_convert_filtered_to_wds.sh
```

Train the decoder-only stage with:

```bash
bash scripts/flowam_finetune_action_decoder.sh
```

Then train LoRA adapters plus the 54-D action head with:

```bash
bash scripts/flowam_finetune_all.sh
```

Or submit the full two-stage 3-day Slurm job with:

```bash
sbatch scripts/flowam_sbatch_2stage.sh
```

Training writes `report-latest.md/json` during evaluation and
`report-final.md/json` at the end of each stage. The FlowAM scripts now default
to 40 epochs per stage, and stage 2 resets training progress counters after
loading stage-1 weights so LoRA gets a full training budget.

The primary report fields are `action_rmse_cm`, `action_cd_cm`,
`scene_rmse_cm`, and `scene_cd_cm`, averaged per scene and reported in
centimeters. For the 54-D FlowAM action vector, the default action CD/RMSE mode
uses the nearest observed robot pointcloud frame in normalized action space and
computes the pointcloud metric on robot surface points. The raw mixed-unit
54-D action-vector errors are still written as `action_vector_rmse` and
`action_vector_mae`.
