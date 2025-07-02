import React, { useState } from 'react';

export default function HamburgerMenu() {
  const [open, setOpen] = useState(false);

  return (
    <div className="hamburger-menu relative z-50">
      <button
        aria-label="Toggle menu"
        className="btn p-3 text-2xl sm:hidden"
        onClick={() => setOpen(!open)}
      >
        &#9776;
      </button>

      {open && (
        <nav className="menu absolute top-12 left-0 w-48 bg-white shadow-lg rounded-lg p-4 flex flex-col gap-2 z-50 border border-gray-200">
          <ul className="flex flex-col gap-2">
            <li><button className="w-full text-left py-2 px-3 rounded hover:bg-gray-100" aria-label="Open gallery">📂 Gallery</button></li>
            <li><button className="w-full text-left py-2 px-3 rounded hover:bg-gray-100" aria-label="Send feedback">💬 Feedback</button></li>
            <li><button className="w-full text-left py-2 px-3 rounded hover:bg-gray-100" aria-label="About the app">ℹ️ About</button></li>
          </ul>
        </nav>
      )}
    </div>
  );
}