function Sidebar() {
  return (
    <div
      style={{
        width: "240px",
        background: "#111827",
        color: "white",
        minHeight: "100vh",
        padding: "20px",
      }}
    >
      <h2>EcoShift AI</h2>

      <ul style={{ marginTop: "30px", listStyle: "none", padding: 0 }}>
        <li style={{ marginBottom: "20px" }}>Dashboard</li>
        <li style={{ marginBottom: "20px" }}>Map Analytics</li>
        <li style={{ marginBottom: "20px" }}>Risk Zones</li>
        <li style={{ marginBottom: "20px" }}>AI Insights</li>
      </ul>
    </div>
  );
}

export default Sidebar;