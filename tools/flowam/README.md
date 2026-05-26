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
`report-final.md/json` at the end of each stage. Scene RMSE/CD are reported in
centimeters. Actuator RMSE is the raw 54-D joint-action RMSE; actuator
point-cloud CD needs FK/surface points and is reported as unavailable.
