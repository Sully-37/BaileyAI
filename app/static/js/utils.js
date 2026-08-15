export function logLatency(event, details = {}) {
    console.info(
        `[Bailey][${new Date().toISOString()}] ${event}`,
        details,
    );
}


export async function readJsonResponse(response) {
    let data;

    try {
        data = await response.json();
    } catch {
        throw new Error(
            `Request failed with HTTP ${response.status}`
        );
    }

    if (!response.ok) {
        const detail = data.detail || data;

        throw new Error(
            detail.message ||
            detail.error ||
            `Request failed with HTTP ${response.status}`
        );
    }

    return data;
}
