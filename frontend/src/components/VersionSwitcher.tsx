import * as React from "react";

export function VersionSwitcher({
  versions,
  defaultVersion,
}: {
  versions: string[];
  defaultVersion: string;
}) {
  return (
    <div className="mb-4">
      <label className="block text-xs font-medium text-muted-foreground mb-1">
        Version
      </label>
      <select className="w-full rounded-md border border-input bg-background px-2 py-1 text-sm">
        {versions.map((v) => (
          <option key={v} value={v} selected={v === defaultVersion}>
            {v}
          </option>
        ))}
      </select>
    </div>
  );
}
