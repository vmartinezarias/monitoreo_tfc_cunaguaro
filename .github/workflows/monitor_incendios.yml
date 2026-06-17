name: Monitor Incendios NASA FIRMS

on:
  schedule:
    # Corre cada 3 horas (NASA FIRMS actualiza cada 3h aprox.)
    - cron: '0 */3 * * *'
  workflow_dispatch:   # permite correrlo manualmente desde la pestaña Actions

jobs:
  monitor-incendios:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - name: Checkout repositorio
        uses: actions/checkout@v4

      - name: Configurar Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      # No requiere dependencias externas — solo stdlib de Python
      - name: Ejecutar monitor de incendios
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
          FIRMS_KEY:    ${{ secrets.FIRMS_KEY }}
        run: python monitor_incendios.py
