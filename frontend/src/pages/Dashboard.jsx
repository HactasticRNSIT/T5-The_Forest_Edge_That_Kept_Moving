import { useEffect, useState } from "react";
import Layout from "../components/Layout";
import StatCard from "../components/StatCard";
import EcoMap from "../maps/EcoMap";

const API = "http://localhost:8000";

function Dashboard() {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/stats`)
      .then(r => r.json())
      .then(setStats)
      .catch(console.error);
  }, []);

  return (
    <Layout>
      <div style={{ display: "flex", gap: "20px", marginBottom: "20px" }}>
        <StatCard title="Degraded Zones" value={stats ? `${stats.degraded_pct}%` : "..."} />
        <StatCard title="At Risk Zones"  value={stats ? stats.at_risk  : "..."} />
        <StatCard title="Stable Zones"   value={stats ? stats.stable   : "..."} />
        <StatCard title="Hidden Stress"  value={stats ? stats.high_stress_zones : "..."} />
      </div>
      <EcoMap />
    </Layout>
  );
}

export default Dashboard;