import vllm.v1.request as req
from unittest.mock import patch

patch.object(req.Request, "num_output_placeholders", 0, create=True).start()