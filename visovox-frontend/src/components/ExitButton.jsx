export default function ExitButton() {
  const handleExit = () => {
    alert("Exiting app or returning to home...");
  };

  return (
    <button onClick={handleExit} className="btn exit-button" aria-label="Exit the app">
      ❌ Exit
    </button>
  );
}