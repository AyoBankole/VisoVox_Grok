export default function ActionButtons({ onAction }) {
  return (
    <div className="action-buttons flex flex-col sm:flex-row flex-wrap gap-2 sm:gap-4 justify-center items-center mt-4" role="group" aria-label="Model action buttons">
      <button className="btn" onClick={() => onAction("read")} aria-label="Read text in image">🔊 Read</button>
      <button className="btn" onClick={() => onAction("ask")} aria-label="Ask a question about image">❓ Ask</button>
      <button className="btn" onClick={() => onAction("caption")} aria-label="Generate caption for image">📝 Caption</button>
    </div>
  );
}