.PHONY: backend web sync-core setup-h100 smoke-realtime

backend:
	cd apps/backend && python -m app.main

web:
	cd apps/web && npm run dev

sync-core:
	./scripts/sync_streamdiffusion_core.sh

setup-h100:
	./scripts/setup_h100_runtime.sh

smoke-realtime:
	cd apps/backend && python scripts/smoke_realtime.py --frames 10 --model wan-1.3b --steps 2 --require-core
