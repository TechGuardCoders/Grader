# Grader — trajectory grading engine container (library image)
# Build:  docker build -t grader .
# Run:    docker run grader    # import smoke: module loads and reports ready

FROM python:3.12-slim AS builder
WORKDIR /app
COPY grader.py .

FROM python:3.12-slim AS runner
WORKDIR /app
COPY --from=builder /app/grader.py .
ENV PYTHONUNBUFFERED=1
CMD ["python", "-c", "import grader; print('grader ready:', grader.__doc__.splitlines()[0])"]
