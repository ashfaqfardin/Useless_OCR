import React, { ReactNode } from "react";
import { cn } from "@/lib/utils";

const navItems = [
  { name: "Playground", href: "#" },
  { name: "History", href: "#" },
  { name: "Starred", href: "#" },
  { name: "Settings", href: "#" },
  { name: "Models", href: "#" },
  { name: "Documentation", href: "#" },
];

export default function SidebarLayout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen bg-background text-foreground">
      {/* Sidebar */}
      <aside className="w-64 bg-muted border-r border-border flex flex-col justify-between py-4 px-3">
        <div>
          <div className="flex items-center gap-2 mb-8 px-2">
            <div className="bg-primary text-primary-foreground rounded-full w-8 h-8 flex items-center justify-center font-bold text-lg">
              E
            </div>
            <div>
              <div className="font-bold leading-tight">EasyOCR</div>
              <div className="text-xs text-muted-foreground">Enterprise</div>
            </div>
          </div>
          <nav className="flex flex-col gap-1">
            {navItems.map((item) => (
              <a
                key={item.name}
                href={item.href}
                className={cn(
                  "flex items-center gap-2 px-3 py-2 rounded hover:bg-accent transition-colors text-sm font-medium text-muted-foreground"
                )}
              >
                {item.name}
              </a>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-2 px-2 py-3 border-t border-border">
          <img
            src="https://github.com/shadcn.png"
            alt="User avatar"
            className="w-8 h-8 rounded-full border"
          />
          <div>
            <div className="font-medium text-sm">shadcn</div>
            <div className="text-xs text-muted-foreground">m@example.com</div>
          </div>
        </div>
      </aside>
      {/* Main content */}
      <main className="flex-1 flex flex-col bg-background p-8 overflow-y-auto">
        {children}
      </main>
    </div>
  );
}
