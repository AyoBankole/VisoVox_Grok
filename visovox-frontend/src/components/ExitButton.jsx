export default function ExitButton() {
  const handleExit = () => {
    alert("Exiting app or returning to home...");
  };

  return (
    <button onClick={handleExit} className="btn exit-button py-3 px-6 text-lg rounded-lg mt-4 w-full sm:w-auto" aria-label="Exit the app">
      ❌ Exit
    </button>
  );
}