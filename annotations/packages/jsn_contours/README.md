# JSN Annotation Package

Use `jsn_contour_manifest.csv` as the source table for annotation import.
The `path` column is repo-local and portable; `local_image_path` resolves to this machine.

No duplicate image folders were created.
If you later want copied package images, rerun with `ANNOTATION_PACKAGE_COPY_IMAGES=1`.

Trace two polylines per image:
- `femoral_surface`
- `tibial_surface`

Export the reviewed result as COCO JSON and place it at:
`annotations/reviewed/jsn_cvat_export.json`