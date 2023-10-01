FROM babashka/babashka:latest

COPY <<EOF pull_model.clj
(ns pull-model
  (:require [clojure.string :as str]
            [cheshire.core :as json]
            [babashka.curl :as http]
            [clojure.java.io :as io]))

;; query Ollama to verify that the model is pulled
;; if model is not pulled, send request to pull it and wait for completion
;; exit with system code -1 if any Ollama api requests fail - Ollama must be running
(try
  (let [llm (get (System/getenv) "LLM")
        url (get (System/getenv) "OLLAMA_BASE_URL")]
    (println (format "pull model %s using %s" llm url))
    ;; verify that this environment is configured ot use Ollama
    (if (and llm url (not (#{"gpt-4" "gpt-3.5"} llm)))
      (let [llm-with-tag (if (str/includes? llm ":") llm (format "%s:latest" llm))
            model (->> (-> (http/get (format "%s/api/tags" url))
                           :body
                           (json/parse-string keyword))
                       :models
                       (filter #(= (:name %) llm-with-tag)))]
        (if (not (seq model))
          (let [pull-response (http/post
                               (format "%s/api/pull" url)
                               {:body (json/generate-string {:name llm-with-tag})
                                :as :stream})
                pull-response-reader (io/reader (:body pull-response))]
            (loop [last ""]
              (when-let [{:keys [status total completed]} (json/parse-stream pull-response-reader keyword)]
                (println
                 (format "%s - %s%s"
                         status
                         (cond (and total completed) (format "%s/%s" completed total) :else "")
                         (if (= last status) "\033[A;[2K" "")))                (recur status))))
          (println (format "found model %s" (json/generate-string (into [] model))))))
      (println "OLLAMA model only pulled if both LLM and OLLAMA_BASE_URL are set and the LLM model is not gpt")))
  (catch Exception e
    (if-let [err (-> (ex-data e) :err)]
      (println "Unable to query Ollama: " err)
      (println e))
    (System/exit 1))
  (catch Throwable t
    (println t)
    (System/exit 1)))
EOF

ENTRYPOINT ["bb", "-f", "pull_model.clj"]

