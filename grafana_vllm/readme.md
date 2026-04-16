steps to have vllm  grafana dashboard:

1. set up Prometheus (promorheus.yml):

```
scrape_configs:
  - job_name: 'vllm'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['host address'] # Your host address
    scheme: https
```

2. start Grafana & Prometheus via Docker

```
docker run -d --name prometheus -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
docker run -d --name grafana -p 3000:3000 grafana/grafana-oss
```

3. Connect them in Grafana

	1. Go to http://localhost:3000 (Login: admin/admin).
	2. Go to Connections > Data Sources > Add Data Source.
	3. Choose Prometheus.
	4. In the URL field, type: http://host.docker.internal:9090 (or the IP of your machine).
	5.Click Save & Test.

4.mport the vLLM Dashboard

	1. In Grafana, click the + (New) > Import.
	2. Enter Dashboard ID: `24756`
	3. Load
	4. Select the "Prometheus" data source you just created and click Import.
