{{- define "rag-api.name" -}}
rag-api
{{- end -}}

{{- define "rag-api.fullname" -}}
{{ include "rag-api.name" . }}
{{- end -}}