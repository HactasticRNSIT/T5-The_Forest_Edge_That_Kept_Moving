function StatCard({ title, value }) {
  return (
    <div
      style={{
        background: "#1f2937",
        color: "white",
        padding: "20px",
        borderRadius: "12px",
        width: "220px",
      }}
    >
      <h3>{title}</h3>
      <h1>{value}</h1>
    </div>
  );
}

export default StatCard;