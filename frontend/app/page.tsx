"use client";

import { useMemo, useRef, useState } from "react";

type AnalyzeResponse =
  | {
      tmh_mm: number;
      diagnosis: string;
      iris_diam_px: number;
      tmh_px_median: number;
    }
  | { error: string };

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const cameraRef = useRef<HTMLInputElement | null>(null);

  const canAnalyze = useMemo(() => !!file && !busy, [file, busy]);

  async function analyze(f: File) {
    setBusy(true);
    setResult(null);
    try {
      const form = new FormData();
      form.append("image", f);
      const res = await fetch(`/api/analyze`, {
        method: "POST",
        body: form,
      });
      const json = (await res.json()) as AnalyzeResponse;
      setResult(json);
    } catch (e) {
      setResult({ error: e instanceof Error ? e.message : "Error desconocido" });
    } finally {
      setBusy(false);
    }
  }

  function onPickFile(nextFile: File | null) {
    setFile(nextFile);
    setResult(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : null);
  }

  async function takePhoto() {
    cameraRef.current?.click();
  }

  return (
    <main className="mx-auto flex min-h-dvh w-full max-w-4xl flex-col gap-6 px-5 py-10">
      <header className="flex flex-col gap-2">
        <div className="text-sm uppercase tracking-wider text-white/60">TMH</div>
        <h1 className="text-3xl font-semibold tracking-tight">Analyzer</h1>
        <p className="text-white/60">
          Sube una imagen (o toma una foto) y calcula el TMH usando tu modelo.
        </p>
      </header>

      <section className="rounded-2xl border bg-white/5 p-5 backdrop-blur">
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              className="rounded-xl bg-white px-4 py-2 text-sm font-medium text-black hover:bg-white/90 disabled:opacity-50"
              onClick={() => inputRef.current?.click()}
              disabled={busy}
            >
              Subir imagen
            </button>
            <button
              type="button"
              className="rounded-xl border px-4 py-2 text-sm font-medium text-white hover:bg-white/5 disabled:opacity-50"
              onClick={takePhoto}
              disabled={busy}
            >
              Tomar foto
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
            />
            <input
              ref={cameraRef}
              type="file"
              accept="image/*"
              capture="environment"
              className="hidden"
              onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
            />
            <div className="text-xs text-white/50">API: /api (rewrite)</div>
          </div>

          {previewUrl ? (
            <div className="overflow-hidden rounded-2xl border bg-black">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={previewUrl} alt="Preview" className="h-auto w-full" />
            </div>
          ) : (
            <div className="rounded-2xl border border-dashed bg-white/5 p-10 text-center text-sm text-white/50">
              No hay imagen seleccionada.
            </div>
          )}

          <div className="flex items-center gap-3">
            <button
              type="button"
              className="rounded-xl bg-emerald-400 px-4 py-2 text-sm font-semibold text-black hover:bg-emerald-300 disabled:opacity-50"
              disabled={!canAnalyze}
              onClick={() => file && analyze(file)}
            >
              {busy ? "Analizando..." : "Calcular TMH"}
            </button>
            <button
              type="button"
              className="rounded-xl border px-4 py-2 text-sm font-medium text-white hover:bg-white/5 disabled:opacity-50"
              disabled={busy && !file}
              onClick={() => onPickFile(null)}
            >
              Limpiar
            </button>
          </div>

          <div className="rounded-2xl border bg-white/5 p-4">
            {result === null ? (
              <div className="text-sm text-white/60">Sin resultados todavía.</div>
            ) : "error" in result ? (
              <div className="text-sm text-red-300">Error: {result.error}</div>
            ) : (
              <div className="grid gap-2">
                <div className="text-sm text-white/60">Resultado</div>
                <div className="text-2xl font-semibold">{result.tmh_mm} mm</div>
                <div className="text-sm text-white/70">{result.diagnosis}</div>
                <div className="text-xs text-white/50">
                  iris_diam_px={result.iris_diam_px} · tmh_px_p25={result.tmh_px_median}
                </div>
              </div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}

