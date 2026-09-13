"use client";

import { useEffect, useRef, useState } from "react";

type Nutrition = {
  calories: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
};

type PredictionResult = {
  food: string;
  confidence: number;
  low_confidence: boolean;
  portion: string;
  weight_g: number;
  nutrition: Nutrition;
};

const PORTIONS = [
  { value: "small", label: "Small" },
  { value: "medium", label: "Medium" },
  { value: "large", label: "Large" },
  { value: "one_person", label: "1 Person" },
  { value: "two_person", label: "2 People" },
];

function formatFoodName(name: string) {
  return name
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatNumber(value: number) {
  return Number.isInteger(value) ? value : Math.round(value * 10) / 10;
}

function confidenceMeta(confidence: number) {
  if (confidence >= 0.75) {
    return { label: "High confidence", color: "#3F6B4B" };
  }
  if (confidence >= 0.5) {
    return { label: "Moderate confidence", color: "#D98E04" };
  }
  return { label: "Low confidence", color: "#B23A2E" };
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [portion, setPortion] = useState("medium");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Revoke the object URL when it's replaced or the component unmounts,
  // so we don't leak memory across selections.
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleFile = (selected: File | null) => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(selected);
    setResult(null);
    setError("");
    setPreview(selected ? URL.createObjectURL(selected) : null);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (!dropped) return;
    if (!dropped.type.startsWith("image/")) {
      setError("That file isn't an image. Please drop a photo of food.");
      return;
    }
    handleFile(dropped);
  };

  const handleReset = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    setPortion("medium");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handlePredict = async () => {
    if (!file) {
      setError("Please select a food image.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("portion", portion);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#FFFFF8] px-4 py-10 sm:px-6 sm:py-16">
      <div className="mx-auto max-w-xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-black tracking-tight text-[#1C1B18] sm:text-4xl">
            NutriVision
          </h1>
          <p className="mt-1 text-sm text-[#5B584F]">
            Photograph a plate, get the nutrition facts.
          </p>
          <div className="mt-5 h-[6px] w-full bg-[#1C1B18]" />
        </div>

        {/* Upload card */}
        <div className="rounded-2xl border border-[#DAD6CC] bg-white p-5 sm:p-7">
          <p className="mb-3 text-sm font-medium text-[#1C1B18]">
            Food image
          </p>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => handleFile(e.target.files?.[0] || null)}
            className="hidden"
          />

          {!preview ? (
            <div
              role="button"
              tabIndex={0}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  fileInputRef.current?.click();
                }
              }}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                setIsDragging(false);
              }}
              onDrop={handleDrop}
              className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 text-center transition-colors ${
                isDragging
                  ? "border-[#D98E04] bg-[#FFF7E6]"
                  : "border-[#DAD6CC] bg-[#FAF9F5] hover:border-[#B7B2A3]"
              }`}
            >
              <svg
                width="34"
                height="34"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#8A8677"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 16V4" />
                <path d="M7 9l5-5 5 5" />
                <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
              </svg>
              <p className="mt-3 text-sm font-medium text-[#1C1B18]">
                Drag and drop a photo here
              </p>
              <p className="mt-1 text-xs text-[#8A8677]">
                or click to browse — JPG or PNG
              </p>
            </div>
          ) : (
            <div>
              <div className="overflow-hidden rounded-xl border border-[#DAD6CC] bg-[#FAF9F5]">
                <img
                  src={preview}
                  alt="Selected food"
                  className="h-64 w-full object-contain"
                />
              </div>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="mt-2 text-xs font-medium text-[#5B584F] underline underline-offset-2 hover:text-[#1C1B18]"
              >
                Choose a different photo
              </button>
            </div>
          )}

          {/* Portion selector */}
          <p className="mb-2 mt-6 text-sm font-medium text-[#1C1B18]">
            Portion size
          </p>
          <div className="flex flex-wrap gap-2">
            {PORTIONS.map((p) => (
              <button
                key={p.value}
                onClick={() => setPortion(p.value)}
                className={`rounded-full border px-4 py-2 text-sm font-medium transition-colors ${
                  portion === p.value
                    ? "border-[#1C1B18] bg-[#1C1B18] text-white"
                    : "border-[#DAD6CC] bg-white text-[#5B584F] hover:border-[#B7B2A3]"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Actions */}
          <div className="mt-7 flex flex-col gap-3 sm:flex-row">
            <button
              onClick={handlePredict}
              disabled={loading || !file}
              className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-[#D98E04] px-6 py-3 font-semibold text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading && (
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
              )}
              {loading ? "Analyzing..." : "Analyze food"}
            </button>

            {(file || result) && !loading && (
              <button
                onClick={handleReset}
                className="rounded-lg border border-[#DAD6CC] px-6 py-3 font-medium text-[#5B584F] hover:border-[#B7B2A3] hover:text-[#1C1B18]"
              >
                Reset
              </button>
            )}
          </div>

          {error && (
            <div className="mt-5 rounded-lg border border-[#E5B9B0] bg-[#FDF2F0] p-4 text-sm text-[#B23A2E]">
              {error}
            </div>
          )}
        </div>

        {/* Loading state */}
        {loading && (
          <div className="mt-6 flex items-center gap-3 rounded-2xl border border-[#DAD6CC] bg-white p-5 text-sm text-[#5B584F]">
            <span className="h-5 w-5 flex-shrink-0 animate-spin rounded-full border-2 border-[#DAD6CC] border-t-[#D98E04]" />
            Reading the plate and matching it against the reference database...
          </div>
        )}

        {/* Result */}
        {result && !loading && (
          <div className="mt-6">
            {result.low_confidence ? (
              <div className="rounded-2xl border border-[#E5B9B0] bg-[#FDF2F0] p-6">
                <p className="font-semibold text-[#B23A2E]">
                  Couldn&apos;t identify this with confidence
                </p>
                <p className="mt-2 text-sm text-[#8A4B44]">
                  Try a clearer, well-lit photo taken straight on, with the
                  food filling most of the frame.
                </p>
                <div className="mt-4 flex items-center justify-between text-sm text-[#8A4B44]">
                  <span>
                    Best guess:{" "}
                    <span className="font-medium">
                      {formatFoodName(result.food)}
                    </span>
                  </span>
                  <span>{(result.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            ) : (
              <div className="rounded-2xl border-2 border-[#1C1B18] bg-white p-6">
                {/* Food name + confidence */}
                <h2 className="text-2xl font-bold text-[#1C1B18]">
                  {formatFoodName(result.food)}
                </h2>

                <div className="mt-2 flex items-center gap-2">
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{
                      backgroundColor: confidenceMeta(result.confidence)
                        .color,
                    }}
                  />
                  <span className="text-xs font-medium text-[#5B584F]">
                    {confidenceMeta(result.confidence).label} ·{" "}
                    {(result.confidence * 100).toFixed(0)}%
                  </span>
                </div>

                {/* Serving size row */}
                <div className="mt-5 flex items-center justify-between border-t border-[#1C1B18] pt-3 text-sm">
                  <span className="text-[#5B584F]">
                    Serving size (
                    {PORTIONS.find((p) => p.value === result.portion)
                      ?.label ?? result.portion}
                    )
                  </span>
                  <span className="font-medium text-[#1C1B18]">
                    {result.weight_g} g
                  </span>
                </div>

                {/* Calories - the big number */}
                <div className="mt-3 flex items-baseline justify-between border-t-[6px] border-[#1C1B18] pt-3">
                  <span className="text-lg font-bold text-[#1C1B18]">
                    Calories
                  </span>
                  <span className="font-mono text-4xl font-black tabular-nums text-[#1C1B18]">
                    {formatNumber(result.nutrition.calories)}
                  </span>
                </div>

                {/* Macros */}
                <div className="mt-1 divide-y divide-[#DAD6CC] border-t border-[#1C1B18]">
                  <MacroRow
                    label="Protein"
                    value={result.nutrition.protein_g}
                  />
                  <MacroRow label="Carbs" value={result.nutrition.carbs_g} />
                  <MacroRow label="Fat" value={result.nutrition.fat_g} />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Footer note */}
        <p className="mt-8 text-center text-xs leading-relaxed text-[#8A8677]">
          Nutrition values are reference estimates for the selected portion
          size, not laboratory measurements — actual values vary with
          recipe, ingredients, and preparation.
        </p>
      </div>
    </main>
  );
}

function MacroRow({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center justify-between py-2.5 text-sm">
      <span className="text-[#5B584F]">{label}</span>
      <span className="font-mono font-semibold tabular-nums text-[#1C1B18]">
        {formatNumber(value)} g
      </span>
    </div>
  );
}