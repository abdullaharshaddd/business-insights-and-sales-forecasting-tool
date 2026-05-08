export default function LoadingSpinner({ message = 'Loading data...' }) {
  return (
    <div className="spinner-container">
      <div className="spinner" />
      <span>{message}</span>
    </div>
  )
}
