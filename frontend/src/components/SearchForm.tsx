import * as React from "react";

export function SearchForm() {
  return (
    <form className="w-full mt-4">
      <input
        type="text"
        placeholder="Search..."
        className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
      />
    </form>
  );
}
