import React, { useState } from 'react';

export default function HamburgerMenu() {
  const [open, setOpen] = useState(false);

  return (
    <div className="hamburger-menu" aria-label="App menu">
      <button
        aria-label="Toggle menu"
        className="btn"
        onClick={() => setOpen(!open)}
      >
        ☰
      </button>

      {open && (
        <nav className="menu" role="navigation" aria-label="Sidebar">
          <ul>
            <li><button aria-label="Open gallery">📂 Gallery</button></li>
            <li><button aria-label="Send feedback">💬 Feedback</button></li>
            <li><button aria-label="About the app">ℹ️ About</button></li>
          </ul>
        </nav>
      )}
    </div>
  );
}