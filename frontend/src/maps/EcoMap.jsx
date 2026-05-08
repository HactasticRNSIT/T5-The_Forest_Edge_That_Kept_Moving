import { useEffect, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const API = "http://localhost:8000";
const COLORS = { degraded: "#ff3e3e", at_risk: "#ff9f43", stable: "#00ff9d" };

function EcoMap() {
  const [zones, setZones] = useState([]);

  useEffect(() => {
    fetch(`${API}/api/zones?limit=1000`)
      .then(r => r.json())
      .then(setZones)
      .catch(console.error);
  }, []);

  return (
    <MapContainer
      center={[12.9716, 77.5946]}
      zoom={11}
      style={{ height: "500px", width: "100%" }}
    >
      <TileLayer
        attribution="© CartoDB"
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />
      {zones.map(z => (
        <CircleMarker
          key={z.cell_id}
          center={[z.center_lat, z.center_lon]}
          radius={4}
          pathOptions={{
            color: COLORS[z.predicted_label] || "#888",
            fillColor: COLORS[z.predicted_label] || "#888",
            fillOpacity: 0.8,
            weight: 0,
          }}
        >
          <Popup>
            <b>{(z.predicted_label || "").toUpperCase()}</b><br />
            NDVI Change: {(z.ndvi_total_change || 0).toFixed(3)}<br />
            Stress: {(z.latent_stress_norm || 0).toFixed(2)}<br />
            Cluster: {z.cluster_name || "–"}
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}

export default EcoMap;