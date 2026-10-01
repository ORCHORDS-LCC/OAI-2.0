def test_r2_kv_binding_protocol_exports_from_knowledge_package() -> None:
    from oai2.knowledge import KvNamespaceBinding as ExportedKvBinding
    from oai2.knowledge import R2BucketBinding as ExportedR2BucketBinding
    from oai2.knowledge import (
        R2ObjectBodyBinding as ExportedR2ObjectBodyBinding,
    )
    from oai2.knowledge.cloudflare_bindings_runtime import (
        KvNamespaceBinding,
        R2BucketBinding,
        R2ObjectBodyBinding,
    )

    assert ExportedR2BucketBinding is R2BucketBinding
    assert ExportedR2ObjectBodyBinding is R2ObjectBodyBinding
    assert ExportedKvBinding is KvNamespaceBinding
