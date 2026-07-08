import io
from email.parser import BytesParser
from email.policy import default
from urllib.parse import parse_qs


class FormField:
    """Small field object compatible with the cgi.FieldStorage usage here."""

    def __init__(self, value="", filename=None, data=b""):
        self.value = value
        self.filename = filename
        self.file = io.BytesIO(data)


class FieldStorage:
    """Minimal multipart/form-data and urlencoded parser for local GUI uploads."""

    def __init__(self, fp, headers, environ):
        self.fields = {}
        content_type = environ.get("CONTENT_TYPE") or headers.get("Content-Type", "")
        content_length = int(headers.get("Content-Length", "0") or "0")
        body = fp.read(content_length) if content_length > 0 else b""

        if content_type.startswith("multipart/form-data"):
            self._parse_multipart(content_type, body)
        else:
            self._parse_urlencoded(body)

    def __contains__(self, key):
        return key in self.fields

    def __getitem__(self, key):
        return self.fields[key]

    def getfirst(self, key, default=None):
        field = self.fields.get(key)
        if field is None:
            return default
        return field.value

    def _parse_multipart(self, content_type, body):
        message_bytes = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n"
            "\r\n"
        ).encode("utf-8") + body
        message = BytesParser(policy=default).parsebytes(message_bytes)

        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue

            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if filename:
                self.fields[name] = FormField(filename=filename, data=payload)
            else:
                charset = part.get_content_charset() or "utf-8"
                value = payload.decode(charset, errors="replace")
                self.fields[name] = FormField(value=value, data=payload)

    def _parse_urlencoded(self, body):
        parsed = parse_qs(body.decode("utf-8", errors="replace"))
        for key, values in parsed.items():
            value = values[0] if values else ""
            self.fields[key] = FormField(value=value, data=value.encode("utf-8"))
