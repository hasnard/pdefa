"""
Batch inference — motor batch desteklemediği için çoklu session kullanır.
"""
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from ._bindings import (
    Context, Model, Session, Tensor,
    prepare_for_engine, PreprocessMeta, load_image_as_tensor,
)


class BatchInference:
    """
    N adet session kullanarak "batch" inference.
    Thread pool ile paralel çalıştırır.

    Kullanım:
        batch = BatchInference(ctx, model, batch_size=4)
        results = batch.run(["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
        for out_tensor, meta in results:
            arr = out_tensor.tolist()      # numpy'siz
    """

    def __init__(self, ctx: Context, model: Model, batch_size: int = 4,
                 num_threads: int = 4):
        self.ctx = ctx
        self.model = model
        self.batch_size = batch_size
        self.num_threads = num_threads

        self._sessions: List[Session] = [
            Session(ctx, model) for _ in range(batch_size)
        ]
        self._executor = ThreadPoolExecutor(max_workers=num_threads)

    def _prep_and_run(self, session: Session, img_path: str,
                      w: int, h: int) -> Tuple[Tensor, PreprocessMeta]:
        tensor, meta = prepare_for_engine(img_path, w, h)
        session.set_input(0, tensor)
        session.run()
        out = session.output(0)        # Tensor — .tolist() ile listeye çevir
        return out, meta

    def run(self, image_paths: List[str],
            target_w: int = 640, target_h: int = 640) -> List[Tuple[Tensor, PreprocessMeta]]:
        """Resim yollarını batch halinde işler."""
        results: List[Optional[Tuple[Tensor, PreprocessMeta]]] = [None] * len(image_paths)

        def worker(i: int):
            session = self._sessions[i % self.batch_size]
            results[i] = self._prep_and_run(session, image_paths[i],
                                            target_w, target_h)

        futures = [self._executor.submit(worker, i)
                   for i in range(len(image_paths))]
        for f in futures:
            f.result()

        return results

    def close(self):
        self._executor.shutdown(wait=True)
        self._sessions.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()