import Layout from "../components/Layout";
import StatCard from "../components/StatCard";
import EcoMap from "../maps/EcoMap";

function Dashboard() {
  return (
    <Layout>
      <div
        style={{
          display: "flex",
          gap: "20px",
          marginBottom: "20px",
        }}
      >
        <StatCard title="Forest Loss" value="32%" />
        <StatCard title="Risk Zones" value="18" />
        <StatCard title="Wetland Stress" value="11" />
      </div>

      <EcoMap />
    </Layout>
  );
}

export default Dashboard;