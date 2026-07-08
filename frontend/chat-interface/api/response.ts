export async function POST(req: Request) {
    const body = await req.text();

    const response = await fetch(
        "http://54.206.87.156:8000/response",
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body
        }
    );

    return new Response(await response.text(), {
        status: response.status,
        headers: {
            "Content-Type": "application/json"
        }
    });
}