import React from "react";

export function SiteHeader() {
  return (
    <header className="w-full h-16 flex items-center justify-between px-6 border-b border-border bg-background/80 backdrop-blur">
      <h1 className="text-lg font-semibold tracking-tight">
        EasyOCR Playground
      </h1>
      <div className="flex items-center gap-4">
        {/* Placeholder for user actions, notifications, etc. */}
        <span className="text-sm text-muted-foreground">User</span>
      </div>
    </header>
  );
}
