# Importers

These readers are unofficial. They follow public export documentation. Encord does not endorse the Encord reader. HumanSignal does not endorse the Label Studio reader. Neither reader talks to a network.

Pass `--format encord` or `--format label-studio`, or let `--format auto` sniff a `.json` file.

## Encord

Basis: the public JSON export description at `https://docs.encord.com/platform-documentation/Annotate/annotate-export/annotate-export-json` (fetched while this package was written; the vendor can change the file).

What is read:

- A list of label rows, or an object with `labels` / `label_rows`, or one label-row object.
- Classifications under `classification_answers`. The selected value is `answers[].name`. In the documented example that is `Night` under the question `Time of day`. The question's own `value` field repeats the question name, so it is not used as the label when `answers` is present.
- `createdBy` (also `created_by`, `lastEditedBy`) as `annotator_id` when it is on the classification. The documented `classification_answers` example does not include `createdBy`. The label row's `label_hash` is the annotator id in that case, which treats one label row as one submission.
- If `classification_answers` yields nothing, classifications nested under `data_units` are tried, including a frame-keyed `labels` object and a direct `classifications` list.

Item ids are `{data_hash}:{question name}` so two questions on one asset are not pooled. `data_title` is the fallback when `data_hash` is absent.

What is not read: bounding boxes, polygons, keypoints, and other geometries. A geometry-only file raises `AnnotrustError` instead of inventing rows.

## Label Studio

Basis: the public JSON export of tasks, each with `annotations` (older files say `completions`) and `result`. See the Label Studio export guide.

What is read:

- `choices` and `taxonomy` results. Several selected choices become one label, joined with `|` in sorted order.
- `completed_by` as an id, a string, or an object. An object uses `email`, then `username`, then `id`.
- `created_at` as the timestamp. Cancelled annotations (`was_cancelled: true`) are skipped.

Item ids are `{task id}:{from_name}`.

What is not read: rectangles, brushes, text areas, and `predictions`.
