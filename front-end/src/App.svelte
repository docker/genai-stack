<script>
    import { tick } from "svelte";
    import SvelteMarkdown from "svelte-markdown";
    import botImage from "./assets/images/bot.jpeg";
    import meImage from "./assets/images/me.jpeg";
    import MdLink from "./lib/MdLink.svelte";

    let messages = [];
    let ragMode = true;
    let question = "How can I create a chatbot on top of my local PDF files using langchain?";
    let shouldAutoScroll = true;
    let input;
    let appState = "idle"; // or receiving
    let senderImages = { bot: botImage, me: meImage };

    async function send() {
        if (!question.trim().length) {
            return;
        }
        appState = "receiving";
        addMessage("me", question, ragMode);
        const messageId = addMessage("bot", "", ragMode);
        try {
            const evt = new EventSource(
                `http://localhost:8504/query-stream?text=${encodeURI(question)}&rag=${ragMode}`
            );
            question = "";
            evt.onmessage = (e) => {
                if (e.data) {
                    const data = JSON.parse(e.data);
                    if (data.init) {
                        updateMessage(messageId, "", data.model);
                        return;
                    }
                    updateMessage(messageId, data.token);
                }
            };
            evt.onerror = (e) => {
                // Stream will end with an error
                // and we want to close the connection on end (otherwise it will keep reconnecting)
                evt.close();
            };
        } catch (e) {
            updateMessage(messageId, "Error: " + e.message);
        } finally {
            appState = "idle";
        }
    }

    function updateMessage(existingId, text, model = null) {
        if (!existingId) {
            return;
        }
        const existingIdIndex = messages.findIndex((m) => m.id === existingId);
        if (existingIdIndex === -1) {
            return;
        }
        messages[existingIdIndex].text += text;
        if (model) {
            messages[existingIdIndex].model = model;
        }
        messages = messages;
    }

    function addMessage(from, text, rag) {
        const newId = Math.random().toString(36).substring(2, 9);
        const message = { id: newId, from, text, rag };
        messages = messages.concat([message]);
        return newId;
    }

    function scrollToBottom(node, _) {
        const scroll = () => node.scrollTo({ top: node.scrollHeight });
        scroll();
        return { update: () => shouldAutoScroll && scroll() };
    }

    function scrolling(e) {
        shouldAutoScroll = e.target.scrollTop + e.target.clientHeight > e.target.scrollHeight - 55;
    }

    $: appState === "idle" && input && focus(input);
    async function focus(node) {
        await tick();
        node.focus();
    }
    // send();
</script>

<main class="h-full text-sm bg-gradient-to-t from-indigo-100 bg-fixed overflow-hidden">
    <div on:scroll={scrolling} class="flex h-full flex-col py-12 overflow-y-auto" use:scrollToBottom={messages}>
        <div class="w-4/5 mx-auto flex flex-col mb-32">
            {#each messages as message (message.id)}
                <div
                    class="max-w-[80%] min-w-[40%] rounded-lg p-4 mb-4 overflow-x-auto bg-white border border-indigo-200"
                    class:self-end={message.from === "me"}
                    class:text-right={message.from === "me"}
                >
                    <div class="flex flex-row items-start gap-2">
                        <div
                            class:ml-auto={message.from === "me"}
                            class="relative w-12 h-12 border border-indigo-200 rounded-lg flex justify-center items-center"
                        >
                            <img
                                src={senderImages[message.from]}
                                alt=""
                                class="w-12 h-12 absolute top-0 left-0 rounded-lg"
                            />
                        </div>
                        {#if message.from === "bot"}
                            <div class="text-sm">
                                <div>Model: {message.model ? message.model : ""}</div>
                                <div>RAG: {message.rag ? "Enabled" : "Disabled"}</div>
                            </div>
                        {/if}
                    </div>
                    <div class="mt-4"><SvelteMarkdown source={message.text} renderers={{ link: MdLink }} /></div>
                </div>
            {/each}
        </div>
        <div class="text-sm w-full fixed bottom-16">
            <div class="shadow-lg bg-indigo-50 rounded-lg w-4/5 xl:w-2/3 2xl:w-1/2 mx-auto">
                <div class="rounded-t-lg px-4 py-2 font-light">
                    <div class="font-semibold">RAG mode</div>
                    <div class="">
                        <label class="mr-2">
                            <input type="radio" bind:group={ragMode} value={false} /> Disabled
                        </label>
                        <label>
                            <input type="radio" bind:group={ragMode} value={true} /> Enabled
                        </label>
                    </div>
                </div>
                <form class="rounded-md w-full bg-white p-2 m-0" on:submit|preventDefault={send}>
                    <input
                        disabled={appState === "receiving"}
                        class="text-lg w-full bg-white focus:outline-none px-4"
                        bind:value={question}
                        bind:this={input}
                        type="text"
                    />
                </form>
            </div>
        </div>
    </div>
</main>

<style>
    :global(pre) {
        @apply bg-gray-100 rounded-lg p-4 border border-indigo-200;
    }
    :global(code) {
        @apply text-indigo-500;
    }
</style>
