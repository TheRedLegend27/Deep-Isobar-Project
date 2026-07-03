"""Excel export for dashboard endpoints — one utility, every table.

Design decision (roadmap Phase 0): any endpoint that returns tabular data
accepts ``?format=xlsx`` and streams a workbook instead of JSON, respecting
the same role checks it already enforces.  Endpoints opt in with one line::

    if format == "xlsx":
        return xlsx_response(df, "trades")

The CFO's P&L workbook, the math guy's sweep results, and the investor
summary all ride this same path — no per-page export code.
"""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
from fastapi.responses import StreamingResponse

_XLSX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


def xlsx_response(
    df: pd.DataFrame,
    basename: str,
    sheet_name: str = "data",
) -> StreamingResponse:
    """Stream *df* as a dated .xlsx download (``basename_YYYY-MM-DD.xlsx``).

    Column widths are sized to content (capped) so the sheet opens readable
    without manual fiddling — these go straight to the CFO/investors.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns, start=1):
            content_width = max(
                [len(str(col))] + [len(str(v)) for v in df[col].head(200)]
            )
            ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = (
                min(content_width + 2, 40)
            )
    buf.seek(0)

    filename = f"{basename}_{date.today().isoformat()}.xlsx"
    return StreamingResponse(
        buf,
        media_type=_XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
