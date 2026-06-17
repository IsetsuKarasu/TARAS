import concurrent.futures


def run_parallel(submit_fn, count, max_workers):
    outputs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(submit_fn) for _ in range(count)]

        for future in concurrent.futures.as_completed(futures):
            try:
                outputs.append(future.result())
            except Exception as exc:
                print(f'Task generated an exception: {exc}')

    return outputs
