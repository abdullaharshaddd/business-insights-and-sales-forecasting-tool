export default function PageHeader({ title, subtitle, children }) {
  return (
    <div className="page-header fade-up">
      <div className="page-header-row">
        <div>
          <h1 className="page-title">{title}</h1>
          {subtitle && <p className="page-subtitle">{subtitle}</p>}
        </div>
        {children && <div>{children}</div>}
      </div>
    </div>
  )
}
